from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_pretrained_cnn import train_validation_fold
from src.utils.config import load_config
from src.utils.metrics import classification_metrics, confusion_matrix_array
from src.utils.plotting import save_confusion_matrix


CYCLIC_VALIDATION_FOLDS = {fold: (fold % 10) + 1 for fold in range(1, 11)}


def _formal_config(config: dict, seed: int, run_name: str) -> dict:
    resolved = copy.deepcopy(config)
    resolved["seed"] = int(seed)
    resolved["run_name"] = str(run_name)
    resolved.setdefault("evaluation", {})["formal_cross_validation"] = True
    resolved["evaluation"]["locked_for_test"] = True
    return resolved


def _backup_summary(source_dir: Path, backup_root: Path | None, run_name: str) -> None:
    if backup_root is None:
        return
    destination = backup_root / run_name / "formal_10fold_summary"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite formal summary backup: {destination}")
    shutil.copytree(source_dir, destination)


def run_cross_validation(
    linear_config: dict,
    finetune_config: dict,
    seeds: list[int],
    backup_root: Path | None = None,
) -> Path:
    if not seeds:
        raise ValueError("At least one fixed seed is required.")
    base_name = str(finetune_config["run_name"])
    summary_root = Path(finetune_config.get("outputs", {}).get("results_dir", "results")) / (
        f"{base_name}_formal_10fold_{len(seeds)}seed"
    )
    if summary_root.exists():
        raise FileExistsError(f"Refusing to overwrite formal cross-validation output: {summary_root}")
    summary_root.mkdir(parents=True)

    fold_rows: list[dict] = []
    aggregate_targets: list[np.ndarray] = []
    aggregate_probabilities: list[np.ndarray] = []
    labels = list(range(int(finetune_config["data"].get("num_classes", 10))))
    for test_fold, val_fold in CYCLIC_VALIDATION_FOLDS.items():
        seed_prediction_paths: list[Path] = []
        for seed in seeds:
            linear_name = f"{base_name}_linear_seed{seed}"
            linear = _formal_config(linear_config, seed=seed, run_name=linear_name)
            linear["evaluation"]["tta"] = {"enabled": False, "offsets_seconds": [0.0]}
            linear_dir = train_validation_fold(
                linear,
                val_fold=val_fold,
                test_fold_override=test_fold,
                backup_root=backup_root,
                evaluate_test=False,
            )

            finetune_name = f"{base_name}_seed{seed}"
            finetune = _formal_config(finetune_config, seed=seed, run_name=finetune_name)
            finetune["training"]["initial_checkpoint_template"] = str(linear_dir / "best_model.pt")
            finetune_dir = train_validation_fold(
                finetune,
                val_fold=val_fold,
                test_fold_override=test_fold,
                backup_root=backup_root,
                evaluate_test=True,
            )
            seed_prediction_paths.append(finetune_dir / "test_predictions.npz")

        targets: np.ndarray | None = None
        probability_sets: list[np.ndarray] = []
        for path in seed_prediction_paths:
            with np.load(path, allow_pickle=False) as payload:
                candidate_targets = payload["targets"]
                if targets is None:
                    targets = candidate_targets
                elif not np.array_equal(targets, candidate_targets):
                    raise ValueError(f"Seed predictions are misaligned for test fold {test_fold}.")
                probability_sets.append(payload["probabilities"])
        assert targets is not None
        probabilities = np.stack(probability_sets, axis=0).mean(axis=0)
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(targets.tolist(), predictions.tolist(), labels)
        row = {
            "test_fold": test_fold,
            "validation_fold": val_fold,
            "seed_count": len(seeds),
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }
        fold_rows.append(row)
        fold_dir = summary_root / f"testfold{test_fold}_valfold{val_fold}"
        fold_dir.mkdir()
        (fold_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(
            fold_dir / "ensemble_predictions.npz",
            targets=targets,
            probabilities=probabilities,
            predictions=predictions,
        )
        save_confusion_matrix(
            confusion_matrix_array(targets.tolist(), predictions.tolist(), labels),
            [str(label) for label in labels],
            fold_dir / "confusion_matrix.png",
            title=f"Test fold {test_fold}",
        )
        aggregate_targets.append(targets)
        aggregate_probabilities.append(probabilities)

    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    summary = {
        "run_name": base_name,
        "protocol": "fixed cyclic validation fold; each UrbanSound8K fold tested exactly once",
        "validation_fold_mapping": CYCLIC_VALIDATION_FOLDS,
        "seeds": [int(seed) for seed in seeds],
        "tta": dict(finetune_config.get("evaluation", {}).get("tta") or {}),
        "folds": fold_rows,
        "test_evaluated": True,
    }
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in fold_rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))

    all_targets = np.concatenate(aggregate_targets)
    all_probabilities = np.concatenate(aggregate_probabilities)
    all_predictions = all_probabilities.argmax(axis=1)
    from sklearn.metrics import f1_score

    summary["per_class_f1"] = [
        float(value)
        for value in f1_score(all_targets, all_predictions, labels=labels, average=None, zero_division=0)
    ]
    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with (summary_root / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fold_rows)
    np.savez_compressed(
        summary_root / "aggregate_predictions.npz",
        targets=all_targets,
        probabilities=all_probabilities,
        predictions=all_predictions,
    )
    save_confusion_matrix(
        confusion_matrix_array(all_targets.tolist(), all_predictions.tolist(), labels),
        [str(label) for label in labels],
        summary_root / "aggregate_confusion_matrix.png",
        title=f"{base_name} formal 10-fold",
    )
    _backup_summary(summary_root, backup_root, summary_root.name)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one locked EfficientAT method over formal UrbanSound8K 10-fold CV.")
    parser.add_argument("--linear-config", required=True)
    parser.add_argument("--finetune-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--backup-root")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_cross_validation(
        load_config(args.linear_config),
        load_config(args.finetune_config),
        seeds=[int(seed) for seed in args.seeds],
        backup_root=Path(args.backup_root) if args.backup_root else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
