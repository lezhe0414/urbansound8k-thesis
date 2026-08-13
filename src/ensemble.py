from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import UrbanSound8KMelDataset
from src.models import build_model
from src.train import _class_names, _device, _val_fold, train_one_fold
from src.utils.config import load_config
from src.utils.metrics import classification_metrics, confusion_matrix_array
from src.utils.plotting import save_confusion_matrix


DEFAULT_SEEDS = (42, 123, 2026)


def normalize_seeds(seeds: list[int] | tuple[int, ...]) -> tuple[int, int, int]:
    normalized = tuple(int(seed) for seed in seeds)
    if len(normalized) != 3:
        raise ValueError("A 3-seed ensemble requires exactly three seeds.")
    if len(set(normalized)) != 3:
        raise ValueError("The three ensemble seeds must be unique.")
    return normalized


def seed_config(base_config: dict, seed: int) -> dict:
    config = copy.deepcopy(base_config)
    base_run_name = str(base_config.get("run_name", base_config["model"]["name"]))
    config["run_name"] = f"{base_run_name}_seed{seed}"
    config["seed"] = int(seed)
    config.setdefault("training", {}).setdefault("ema", {})["enabled"] = False
    config.setdefault("evaluation", {})["run_test"] = False
    return config


def average_probabilities(probabilities: list[torch.Tensor]) -> torch.Tensor:
    if not probabilities:
        raise ValueError("At least one probability tensor is required.")
    reference_shape = probabilities[0].shape
    if any(probability.shape != reference_shape for probability in probabilities[1:]):
        raise ValueError("All probability tensors must have the same shape.")
    return torch.stack(probabilities, dim=0).mean(dim=0)


def _validate_existing_seed_run(run_dir: Path, expected_config: dict, fold: int) -> None:
    required = [
        run_dir / "best_model.pt",
        run_dir / "config_resolved.json",
        run_dir / "validation_metrics.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete seed run {run_dir}; missing: {missing}")

    resolved = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    config = resolved["config"]
    seed = int(expected_config["seed"])
    if int(resolved["fold"]) != fold or int(config.get("seed", -1)) != seed:
        raise ValueError(f"Existing seed run does not match seed={seed}, fold={fold}: {run_dir}")
    if config != expected_config:
        raise ValueError(
            f"Existing seed run configuration differs from the locked ensemble configuration: {run_dir}"
        )
    if bool(config.get("training", {}).get("ema", {}).get("enabled", False)):
        raise ValueError(f"EMA must be disabled for the seed ensemble: {run_dir}")
    if bool(config.get("evaluation", {}).get("run_test", True)):
        raise ValueError(f"Individual seed runs must be validation-only: {run_dir}")


def _train_or_reuse_seed(config: dict, fold: int, skip_existing: bool) -> Path:
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results"))
    run_dir = results_dir / f"{config['run_name']}_fold{fold}"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not skip_existing:
            raise FileExistsError(
                f"Seed run already exists: {run_dir}. Use --skip-existing only after verifying it belongs to this protocol."
            )
        _validate_existing_seed_run(run_dir, config, fold)
        print(f"Reusing validation-only seed run: {run_dir}")
        return run_dir
    return train_one_fold(config, fold)


def _load_models(run_dirs: list[Path], device: torch.device) -> tuple[list[torch.nn.Module], dict]:
    models: list[torch.nn.Module] = []
    reference_config: dict | None = None
    for run_dir in run_dirs:
        resolved = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
        config = resolved["config"]
        model = build_model(config).to(device)
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models.append(model)
        if reference_config is None:
            reference_config = config
    if reference_config is None:
        raise ValueError("No seed runs were provided.")
    return models, reference_config


def _ensemble_predictions(
    models: list[torch.nn.Module],
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], np.ndarray, float]:
    y_true: list[int] = []
    y_pred: list[int] = []
    probability_batches: list[np.ndarray] = []
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            model_probabilities = [F.softmax(model(inputs), dim=1) for model in models]
            probabilities = average_probabilities(model_probabilities)
            loss = F.nll_loss(probabilities.clamp_min(1e-12).log(), targets)
            total_loss += float(loss.item()) * inputs.size(0)
            predictions = probabilities.argmax(dim=1)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            probability_batches.append(probabilities.cpu().numpy())

    stacked_probabilities = np.concatenate(probability_batches, axis=0)
    return y_true, y_pred, stacked_probabilities, total_loss / max(len(loader.dataset), 1)


def _evaluate_ensemble_split(
    models: list[torch.nn.Module],
    config: dict,
    fold: int,
    split: str,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int], np.ndarray]:
    data_config = config["data"]
    training_config = config["training"]
    val_fold = _val_fold(fold, int(data_config.get("val_fold_offset", 1)))
    max_samples = data_config.get(f"max_{split}_samples")
    dataset = UrbanSound8KMelDataset(
        data_config["processed_dir"],
        split=split,
        test_fold=fold,
        val_fold=val_fold,
        max_samples=max_samples,
        preload=bool(data_config.get("preload", False)),
        feature_representation=str(data_config.get("feature_representation", "mel")),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    y_true, y_pred, probabilities, loss = _ensemble_predictions(models, loader, device)
    labels = list(range(int(data_config.get("num_classes", 10))))
    metrics = classification_metrics(y_true, y_pred, labels)
    metrics[f"{split}_loss"] = loss
    return metrics, y_true, y_pred, probabilities


def _write_seed_summary(run_dirs: list[Path], seeds: tuple[int, int, int], output_path: Path) -> list[dict]:
    rows: list[dict] = []
    for seed, run_dir in zip(seeds, run_dirs, strict=True):
        metrics = json.loads((run_dir / "validation_metrics.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "seed": seed,
                "run_name": run_dir.name,
                "best_epoch": int(metrics["best_epoch"]),
                "val_accuracy": float(metrics["val_accuracy"]),
                "val_f1_macro": float(metrics["val_f1_macro"]),
                "val_loss": float(metrics["val_loss"]),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_seed_ensemble(
    config_path: Path,
    fold: int,
    seeds: tuple[int, int, int],
    skip_existing: bool,
    run_test: bool,
) -> Path:
    base_config = load_config(config_path)
    base_run_name = str(base_config.get("run_name", base_config["model"]["name"]))
    ensemble_name = f"{base_run_name}_3seed"
    results_dir = Path(base_config.get("outputs", {}).get("results_dir", "results"))
    figures_dir = Path(base_config.get("outputs", {}).get("figures_dir", "figures"))
    ensemble_dir = results_dir / f"{ensemble_name}_fold{fold}"
    metrics_path = ensemble_dir / "metrics.json"

    if ensemble_dir.exists() and any(ensemble_dir.iterdir()) and not skip_existing:
        raise FileExistsError(f"Ensemble run already exists: {ensemble_dir}. Use --skip-existing to resume it.")
    if run_test and metrics_path.exists():
        raise FileExistsError(
            f"Final ensemble test metrics already exist: {metrics_path}. Refusing to evaluate the test set twice."
        )
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    seed_run_dirs = [
        _train_or_reuse_seed(seed_config(base_config, seed), fold, skip_existing)
        for seed in seeds
    ]
    seed_rows = _write_seed_summary(seed_run_dirs, seeds, ensemble_dir / "seed_validation_summary.csv")

    device = _device(str(base_config["training"].get("device", "auto")))
    models, model_config = _load_models(seed_run_dirs, device)
    validation_metrics, val_true, val_pred, val_probabilities = _evaluate_ensemble_split(
        models, model_config, fold, "val", device
    )
    (ensemble_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        ensemble_dir / "validation_predictions.npz",
        targets=np.asarray(val_true, dtype=np.int64),
        predictions=np.asarray(val_pred, dtype=np.int64),
        probabilities=val_probabilities,
    )

    protocol = {
        "base_config": str(config_path),
        "ensemble_name": ensemble_name,
        "fold": fold,
        "val_fold": _val_fold(fold, int(base_config["data"].get("val_fold_offset", 1))),
        "seeds": list(seeds),
        "seed_run_dirs": [str(path) for path in seed_run_dirs],
        "checkpoint_selection": "highest validation Macro F1 independently for each seed",
        "aggregation": "arithmetic mean of softmax probabilities",
        "ema_enabled": False,
        "individual_seed_test_evaluation": False,
        "test_evaluated": False,
    }
    protocol_path = ensemble_dir / "ensemble_protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"seed_validation": seed_rows, "ensemble_validation": validation_metrics}, indent=2))
    if not run_test:
        print(f"Wrote validation-only ensemble outputs to {ensemble_dir}; test evaluation was skipped")
        return ensemble_dir

    test_metrics, test_true, test_pred, test_probabilities = _evaluate_ensemble_split(
        models, model_config, fold, "test", device
    )
    metrics_path.write_text(json.dumps(test_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        ensemble_dir / "test_predictions.npz",
        targets=np.asarray(test_true, dtype=np.int64),
        predictions=np.asarray(test_pred, dtype=np.int64),
        probabilities=test_probabilities,
    )
    labels = list(range(int(base_config["data"].get("num_classes", 10))))
    matrix = confusion_matrix_array(test_true, test_pred, labels)
    matrix_path = figures_dir / f"{ensemble_name}_fold{fold}_confusion_matrix.png"
    save_confusion_matrix(
        matrix,
        _class_names(Path(base_config["data"]["processed_dir"])),
        matrix_path,
        title=f"{ensemble_name} fold {fold}",
    )
    protocol["test_evaluated"] = True
    protocol["test_metrics_path"] = str(metrics_path)
    protocol["confusion_matrix_path"] = str(matrix_path)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ensemble_test": test_metrics}, indent=2, sort_keys=True))
    print(f"Wrote final ensemble outputs to {ensemble_dir}")
    print(f"Wrote confusion matrix to {matrix_path}")
    return ensemble_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a validation-selected 3-seed CNN ensemble.")
    parser.add_argument("--config", default="configs/cnn_aug_final.yaml", help="Locked CNN configuration.")
    parser.add_argument("--fold", type=int, default=10, help="UrbanSound8K test fold (1-10).")
    parser.add_argument("--seeds", type=int, nargs=3, default=DEFAULT_SEEDS, metavar=("SEED1", "SEED2", "SEED3"))
    parser.add_argument("--skip-existing", action="store_true", help="Reuse complete validation-only seed runs.")
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Evaluate the fixed probability ensemble on the test fold exactly once.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fold < 1 or args.fold > 10:
        raise ValueError("--fold must be between 1 and 10")
    run_seed_ensemble(
        Path(args.config),
        args.fold,
        normalize_seeds(args.seeds),
        args.skip_existing,
        args.run_test,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
