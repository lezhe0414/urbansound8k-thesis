from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.data import UrbanSound8KWaveformDataset
from src.models.pretrained_efficientat import PretrainedEfficientATClassifier
from src.utils.config import load_config
from src.utils.metrics import classification_metrics, confusion_matrix_array, write_history_csv
from src.utils.plotting import save_confusion_matrix, save_training_history
from src.utils.seed import set_seed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - PyTorch 2.2 compatibility
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _class_names(raw_dir: Path) -> list[str]:
    import pandas as pd

    metadata = pd.read_csv(raw_dir / "metadata" / "UrbanSound8K.csv")
    classes = metadata[["classID", "class"]].drop_duplicates().sort_values("classID")
    return [str(row["class"]) for row in classes.to_dict("records")]


def _class_weights(dataset: UrbanSound8KWaveformDataset, labels: list[int], device: torch.device, config: dict):
    if not bool(config.get("enabled", False)):
        return None
    counts = torch.zeros(len(labels), dtype=torch.float32)
    label_to_index = {label: index for index, label in enumerate(labels)}
    for item in dataset.items:
        counts[label_to_index[item.class_id]] += 1.0
    weights = counts.sum() / (len(labels) * counts.clamp_min(1.0))
    weights = weights.pow(float(config.get("power", 1.0)))
    return (weights / weights.mean()).to(device)


def _class_aware_sampler(dataset: UrbanSound8KWaveformDataset, labels: list[int], config: dict):
    if not bool(config.get("enabled", False)):
        return None
    counts = torch.zeros(len(labels), dtype=torch.float32)
    label_to_index = {label: index for index, label in enumerate(labels)}
    class_indices: list[int] = []
    for item in dataset.items:
        class_index = label_to_index[item.class_id]
        counts[class_index] += 1.0
        class_indices.append(class_index)
    class_weights = (1.0 / counts.clamp_min(1.0)).pow(float(config.get("power", 1.0)))
    sample_weights = torch.tensor([float(class_weights[index]) for index in class_indices], dtype=torch.float32)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _run_epoch(
    model: PretrainedEfficientATClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    labels: list[int],
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    mixed_precision: bool = False,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for waveforms, targets in tqdm(loader, desc="train" if training else "validation", leave=False):
        waveforms = waveforms.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                logits = model(waveforms)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += float(loss.detach().item()) * waveforms.size(0)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(logits.detach().argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred, labels)
    return total_loss / max(len(loader.dataset), 1), metrics


def _backup_run(run_dir: Path, backup_root: Path | None, run_name: str) -> Path | None:
    if backup_root is None:
        return None
    destination = backup_root / run_name / run_dir.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing Drive backup: {destination}")
    shutil.copytree(run_dir, destination)
    return destination


def train_validation_fold(
    config: dict,
    val_fold: int,
    backup_root: Path | None = None,
    evaluate_test: bool = False,
) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    data_config = config["data"]
    model_config = dict(config["model"])
    training_config = config["training"]
    output_config = config.get("outputs", {})
    evaluation_config = config.get("evaluation", {})

    test_fold = int(data_config.get("sealed_test_fold", 10))
    development_folds = [int(value) for value in data_config.get("development_folds", [1, 4, 7])]
    if test_fold != 10:
        raise ValueError("This study requires fold 10 to remain the sealed test fold.")
    if val_fold not in development_folds:
        raise ValueError(f"val_fold={val_fold} is not one of the configured development folds {development_folds}.")
    if evaluate_test and not bool(evaluation_config.get("locked_for_test", False)):
        raise PermissionError("Test evaluation requires evaluation.locked_for_test=true after one final config is locked.")

    raw_dir = Path(data_config["raw_dir"])
    waveform_cache_dir = data_config.get("waveform_cache_dir")
    labels = list(range(int(data_config.get("num_classes", 10))))
    class_names = _class_names(raw_dir)
    run_name = str(config["run_name"])
    run_dir = Path(output_config.get("results_dir", "results")) / run_name / f"valfold{val_fold}"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {
        "raw_dir": raw_dir,
        "test_fold": test_fold,
        "val_fold": val_fold,
        "sample_rate": int(data_config.get("sample_rate", 32_000)),
        "clip_duration_seconds": float(data_config.get("clip_duration_seconds", 5.0)),
        "waveform_cache_dir": waveform_cache_dir,
        "require_cache": bool(data_config.get("require_waveform_cache", False)),
    }
    train_set = UrbanSound8KWaveformDataset(
        split="train",
        max_samples=data_config.get("max_train_samples"),
        **dataset_kwargs,
    )
    val_set = UrbanSound8KWaveformDataset(
        split="val",
        max_samples=data_config.get("max_val_samples"),
        **dataset_kwargs,
    )
    sampler = _class_aware_sampler(train_set, labels, training_config.get("class_aware_sampling", {}))
    workers = int(training_config.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": int(training_config.get("batch_size", 32)),
        "num_workers": workers,
        "pin_memory": bool(training_config.get("pin_memory", True)) and torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        shuffle=sampler is None,
        sampler=sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    device = _device(str(training_config.get("device", "auto")))
    model_name = model_config.pop("name")
    if model_name != "pretrained_efficientat":
        raise ValueError(f"Expected model.name=pretrained_efficientat, received {model_name}")
    model = PretrainedEfficientATClassifier(**model_config).to(device)
    initial_checkpoint = None
    initial_checkpoint_template = training_config.get("initial_checkpoint_template")
    if initial_checkpoint_template:
        initial_checkpoint = Path(str(initial_checkpoint_template).format(val_fold=val_fold))
        if not initial_checkpoint.exists():
            raise FileNotFoundError(f"Missing required linear-probe checkpoint: {initial_checkpoint}")
        try:
            initial_payload = torch.load(initial_checkpoint, map_location=device, weights_only=False)
        except TypeError:  # pragma: no cover - compatibility with older supported PyTorch
            initial_payload = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(initial_payload["model_state"])
        model.set_training_stage(model.stage, partial_last_blocks=model.partial_last_blocks)
    parameter_counts = model.parameter_counts()
    class_weights = _class_weights(train_set, labels, device, training_config.get("class_weighting", {}))
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(training_config.get("label_smoothing", 0.0)),
    )
    encoder_lr = float(training_config.get("encoder_learning_rate", 1e-5))
    head_lr = float(training_config.get("head_learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(encoder_lr=encoder_lr, head_lr=head_lr),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    epochs = int(training_config.get("epochs", 5))
    mixed_precision = bool(training_config.get("mixed_precision", True))
    scaler = _grad_scaler(mixed_precision and device.type == "cuda")
    scheduler_name = str(training_config.get("scheduler", {}).get("name", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported pretrained CNN scheduler: {scheduler_name}")

    start_time = _utc_now()
    start_clock = time.perf_counter()
    history: list[dict] = []
    best_f1 = -1.0
    best_path = run_dir / "best_model.pt"
    for epoch in range(1, epochs + 1):
        train_loss, train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            labels,
            optimizer=optimizer,
            scaler=scaler,
            mixed_precision=mixed_precision,
        )
        val_loss, val_metrics = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            labels,
            mixed_precision=mixed_precision,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "encoder_learning_rate": next(
                (group["lr"] for group in optimizer.param_groups if group.get("group_name") == "encoder"),
                0.0,
            ),
            "head_learning_rate": next(
                group["lr"] for group in optimizer.param_groups if group.get("group_name") == "head"
            ),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "val_fold": val_fold,
                    "sealed_test_fold": test_fold,
                    "best_epoch": epoch,
                    "selected_by": "validation_macro_f1",
                    "checkpoint_sha256": model.checkpoint_sha256,
                },
                best_path,
            )
        if scheduler is not None:
            scheduler.step()

    duration_seconds = time.perf_counter() - start_clock
    end_time = _utc_now()
    write_history_csv(history, run_dir / "history.csv")
    save_training_history(history, run_dir / "training_history.png", title=f"{run_name} validation fold {val_fold}")
    best_row = max(history, key=lambda item: float(item["val_f1_macro"]))
    validation_metrics = {
        "best_epoch": int(best_row["epoch"]),
        "train_accuracy": float(best_row["train_accuracy"]),
        "train_f1_macro": float(best_row["train_f1_macro"]),
        "train_loss": float(best_row["train_loss"]),
        "val_accuracy": float(best_row["val_accuracy"]),
        "val_f1_macro": float(best_row["val_f1_macro"]),
        "val_loss": float(best_row["val_loss"]),
        "selection_metric": "validation_macro_f1",
    }
    (run_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "run_name": run_name,
        "model_name": "EfficientAT MN10",
        "pretrained_checkpoint": model.checkpoint_url,
        "checkpoint_sha256": model.checkpoint_sha256,
        "upstream_commit": "a425fdce92572e602a1d5634799bd9f1f2efa806",
        "stage": model.stage,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint is not None else None,
        "preprocessing": {
            "sample_rate": dataset_kwargs["sample_rate"],
            "clip_duration_seconds": dataset_kwargs["clip_duration_seconds"],
            "feature_extraction": "EfficientAT AugmentMelSTFT",
            "n_fft": model_config.get("n_fft", 1024),
            "win_length": model_config.get("win_length", 800),
            "hop_size": model_config.get("hop_size", 320),
            "n_mels": model_config.get("n_mels", 128),
            "normalization": "(log_mel + 4.5) / 5",
        },
        "parameter_counts": parameter_counts,
        "development_validation_fold": val_fold,
        "sealed_test_fold": test_fold,
        "test_evaluated": False,
        "seed": seed,
        "epochs": epochs,
        "optimizer": "AdamW",
        "encoder_learning_rate": encoder_lr if model.stage == "partial_finetune" else 0.0,
        "head_learning_rate": head_lr,
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "duration_seconds": duration_seconds,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config_resolved.json").write_text(
        json.dumps({"config": config, "val_fold": val_fold}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if evaluate_test:
        test_set = UrbanSound8KWaveformDataset(split="test", **dataset_kwargs)
        test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_loss, test_metrics = _run_epoch(
            model,
            test_loader,
            criterion,
            device,
            labels,
            mixed_precision=mixed_precision,
        )
        test_metrics["test_loss"] = test_loss
        (run_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Re-run predictions only when the uniquely locked test evaluation is explicitly requested.
        y_true: list[int] = []
        y_pred: list[int] = []
        model.eval()
        with torch.no_grad():
            for waveforms, targets in tqdm(test_loader, desc="test", leave=False):
                logits = model(waveforms.to(device, non_blocking=True))
                y_true.extend(targets.tolist())
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        matrix = confusion_matrix_array(y_true, y_pred, labels)
        save_confusion_matrix(matrix, class_names, run_dir / "test_confusion_matrix.png", title=run_name)
        manifest["test_evaluated"] = True
        (run_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    backup_path = _backup_run(run_dir, backup_root, run_name)
    print(f"Wrote pretrained CNN outputs to {run_dir}")
    if backup_path is not None:
        print(f"Backed up run to {backup_path}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EfficientAT on a sealed UrbanSound8K development fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-fold", required=True, type=int)
    parser.add_argument("--backup-root")
    parser.add_argument("--evaluate-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    train_validation_fold(
        config,
        val_fold=args.val_fold,
        backup_root=Path(args.backup_root) if args.backup_root else None,
        evaluate_test=args.evaluate_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
