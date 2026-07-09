from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import UrbanSound8KMelDataset
from src.models import build_model
from src.utils.config import load_config
from src.utils.metrics import classification_metrics, confusion_matrix_array, write_history_csv
from src.utils.plotting import save_confusion_matrix
from src.utils.seed import set_seed


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _val_fold(test_fold: int, offset: int) -> int:
    return ((test_fold - 1 + offset) % 10) + 1


def _class_names(processed_dir: Path) -> list[str]:
    import pandas as pd

    metadata = pd.read_csv(processed_dir / "metadata.csv")
    classes = metadata[["classID", "class"]].drop_duplicates().sort_values("classID")
    return [str(row["class"]) for row in classes.to_dict("records")]


def _run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for inputs, targets in tqdm(loader, desc="train" if training else "eval", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * inputs.size(0)
        predictions = logits.argmax(dim=1)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(predictions.detach().cpu().tolist())

    return total_loss / max(len(loader.dataset), 1), y_true, y_pred


def train_one_fold(config: dict, fold: int) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config["data"]
    training_config = config["training"]
    outputs_config = config.get("outputs", {})
    processed_dir = Path(data_config["processed_dir"])
    val_fold = _val_fold(fold, int(data_config.get("val_fold_offset", 1)))
    labels = list(range(int(data_config.get("num_classes", 10))))
    class_names = _class_names(processed_dir)

    run_name = str(config.get("run_name", config["model"]["name"]))
    results_dir = Path(outputs_config.get("results_dir", "results"))
    figures_dir = Path(outputs_config.get("figures_dir", "figures"))
    run_dir = results_dir / f"{run_name}_fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preload = bool(data_config.get("preload", False))
    max_train_samples = data_config.get("max_train_samples")
    max_val_samples = data_config.get("max_val_samples")
    max_test_samples = data_config.get("max_test_samples")

    train_set = UrbanSound8KMelDataset(
        processed_dir,
        split="train",
        test_fold=fold,
        val_fold=val_fold,
        max_samples=max_train_samples,
        preload=preload,
    )
    val_set = UrbanSound8KMelDataset(
        processed_dir,
        split="val",
        test_fold=fold,
        val_fold=val_fold,
        max_samples=max_val_samples,
        preload=preload,
    )
    test_set = UrbanSound8KMelDataset(
        processed_dir,
        split="test",
        test_fold=fold,
        val_fold=val_fold,
        max_samples=max_test_samples,
        preload=preload,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    device = _device(str(training_config.get("device", "auto")))
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(training_config.get("label_smoothing", 0.0)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )
    epochs = int(training_config.get("epochs", 10))
    scheduler_config = training_config.get("scheduler", {})
    scheduler_name = str(scheduler_config.get("name", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(scheduler_config.get("min_learning_rate", 0.0)),
        )
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 1)),
        )

    history: list[dict] = []
    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_true, val_pred = _run_epoch(model, val_loader, criterion, device)
        train_metrics = classification_metrics(train_true, train_pred, labels)
        val_metrics = classification_metrics(val_true, val_pred, labels)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({"model_state": model.state_dict(), "config": config, "fold": fold, "val_fold": val_fold}, best_path)
        if scheduler is not None:
            if scheduler_name == "reduce_on_plateau":
                scheduler.step(val_metrics["f1_macro"])
            else:
                scheduler.step()

    torch.save({"model_state": model.state_dict(), "config": config, "fold": fold, "val_fold": val_fold}, run_dir / "last_model.pt")
    write_history_csv(history, run_dir / "history.csv")
    (run_dir / "config_resolved.json").write_text(
        json.dumps({"config": config, "fold": fold, "val_fold": val_fold}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_true, test_pred = _run_epoch(model, test_loader, criterion, device)
    test_metrics = classification_metrics(test_true, test_pred, labels)
    test_metrics["test_loss"] = test_loss
    (run_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    matrix = confusion_matrix_array(test_true, test_pred, labels)
    matrix_path = figures_dir / f"{run_name}_fold{fold}_confusion_matrix.png"
    save_confusion_matrix(matrix, class_names, matrix_path, title=f"{run_name} fold {fold}")
    print(f"Wrote run outputs to {run_dir}")
    print(f"Wrote confusion matrix to {matrix_path}")
    return run_dir


def summarize_cross_validation(run_dirs: list[Path], run_name: str, results_dir: Path) -> Path:
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "test_loss"]
    rows: list[dict[str, float | int]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        fold = int(run_dir.name.rsplit("_fold", 1)[1])
        row: dict[str, float | int] = {"fold": fold}
        for key in metric_keys:
            row[key] = float(metrics[key])
        rows.append(row)

    rows = sorted(rows, key=lambda item: int(item["fold"]))
    summary: dict[str, object] = {
        "run_name": run_name,
        "folds": rows,
        "mean": {},
        "std": {},
    }
    for key in metric_keys:
        values = torch.tensor([float(row[key]) for row in rows], dtype=torch.float32)
        summary["mean"][key] = float(values.mean().item())
        summary["std"][key] = float(values.std(unbiased=False).item())

    summary_path = results_dir / f"{run_name}_10fold_summary.json"
    csv_path = results_dir / f"{run_name}_10fold_summary.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", *metric_keys])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote 10-fold summary to {summary_path}")
    print(f"Wrote 10-fold CSV to {csv_path}")
    print(json.dumps({"mean": summary["mean"], "std": summary["std"]}, indent=2, sort_keys=True))
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UrbanSound8K spectrogram classifiers.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--fold", default="10", help="UrbanSound8K test fold: 1-10 or all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_name = str(config.get("run_name", config["model"]["name"]))
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results"))
    if args.fold == "all":
        run_dirs = []
        for fold in range(1, 11):
            run_dirs.append(train_one_fold(config, fold))
        summarize_cross_validation(run_dirs, run_name, results_dir)
    else:
        fold = int(args.fold)
        if fold < 1 or fold > 10:
            raise ValueError("--fold must be 1-10 or all")
        train_one_fold(config, fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
