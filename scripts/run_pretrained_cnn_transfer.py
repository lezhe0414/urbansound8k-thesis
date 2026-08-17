from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_pretrained_cnn import train_validation_fold
from src.utils.config import load_config


def summarize(config: dict, run_dirs: list[Path]) -> tuple[Path, Path]:
    rows: list[dict[str, float | int | str]] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "validation_metrics.json"
        manifest_path = run_dir / "experiment_manifest.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": str(config["run_name"]),
                "validation_fold": int(manifest["development_validation_fold"]),
                "best_epoch": int(metrics["best_epoch"]),
                "validation_macro_f1": float(metrics["val_f1_macro"]),
                "validation_accuracy": float(metrics["val_accuracy"]),
                "validation_loss": float(metrics["val_loss"]),
                "training_time_seconds": float(manifest["duration_seconds"]),
                "test_evaluated": bool(manifest["test_evaluated"]),
            }
        )
    rows.sort(key=lambda item: int(item["validation_fold"]))
    if any(bool(row["test_evaluated"]) for row in rows):
        raise ValueError("Development summary must not include test-evaluated runs.")

    f1_values = np.asarray([float(row["validation_macro_f1"]) for row in rows], dtype=np.float64)
    accuracy_values = np.asarray([float(row["validation_accuracy"]) for row in rows], dtype=np.float64)
    summary = {
        "run_name": str(config["run_name"]),
        "selection_metric": "development_validation_macro_f1_mean",
        "sealed_test_fold": int(config["data"]["sealed_test_fold"]),
        "development_folds": [int(row["validation_fold"]) for row in rows],
        "folds": rows,
        "validation_macro_f1_mean": float(f1_values.mean()),
        "validation_macro_f1_std": float(f1_values.std(ddof=0)),
        "validation_accuracy_mean": float(accuracy_values.mean()),
        "validation_accuracy_std": float(accuracy_values.std(ddof=0)),
        "control_macro_f1_mean": 0.7818,
        "macro_f1_delta_vs_control": float(f1_values.mean() - 0.7818),
        "test_evaluated": False,
    }
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results")) / str(config["run_name"])
    json_path = results_dir / "development_summary.json"
    csv_path = results_dir / "development_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return json_path, csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EfficientAT transfer learning on development folds only.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--backup-root")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    configured_folds = [int(value) for value in config["data"].get("development_folds", [1, 4, 7])]
    folds = args.folds or configured_folds
    if sorted(folds) != sorted(configured_folds):
        raise ValueError(f"This study must run all configured development folds: {configured_folds}")

    results_root = Path(config.get("outputs", {}).get("results_dir", "results")) / str(config["run_name"])
    run_dirs: list[Path] = []
    for fold in folds:
        run_dir = results_root / f"valfold{fold}"
        if args.skip_existing and (run_dir / "validation_metrics.json").exists():
            print(f"Skipping completed run: {run_dir}")
        else:
            run_dir = train_validation_fold(
                config,
                val_fold=fold,
                backup_root=Path(args.backup_root) if args.backup_root else None,
                evaluate_test=False,
            )
        run_dirs.append(run_dir)
    json_path, csv_path = summarize(config, run_dirs)

    if args.backup_root:
        backup_summary_dir = Path(args.backup_root) / str(config["run_name"])
        backup_summary_dir.mkdir(parents=True, exist_ok=True)
        for source in (json_path, csv_path):
            destination = backup_summary_dir / source.name
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing Drive summary: {destination}")
            destination.write_bytes(source.read_bytes())
        print(f"Backed up development summary to {backup_summary_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
