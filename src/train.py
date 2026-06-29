from __future__ import annotations

import argparse
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 0.001)),
        weight_decay=float(training_config.get("weight_decay", 0.0001)),
    )

    history: list[dict] = []
    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"
    epochs = int(training_config.get("epochs", 10))
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
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save({"model_state": model.state_dict(), "config": config, "fold": fold, "val_fold": val_fold}, best_path)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UrbanSound8K spectrogram classifiers.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--fold", default="10", help="UrbanSound8K test fold: 1-10 or all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.fold == "all":
        for fold in range(1, 11):
            train_one_fold(config, fold)
    else:
        fold = int(args.fold)
        if fold < 1 or fold > 10:
            raise ValueError("--fold must be 1-10 or all")
        train_one_fold(config, fold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
