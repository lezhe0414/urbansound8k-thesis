from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


DEFAULT_CONFIGS = [
    "configs/cnn_aug_control.yaml",
    "configs/cnn_aug_light.yaml",
    "configs/cnn_aug_balanced.yaml",
    "configs/cnn_aug_strong.yaml",
]
SUMMARY_KEYS = [
    "best_epoch",
    "train_accuracy",
    "train_f1_macro",
    "val_accuracy",
    "val_f1_macro",
    "val_loss",
]


def _run_name(config_path: Path) -> str:
    config = load_config(config_path)
    return str(config.get("run_name", config["model"]["name"]))


def _run_dir(root: Path, config_path: Path, fold: int) -> Path:
    config = load_config(config_path)
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    return results_dir / f"{_run_name(config_path)}_fold{fold}"


def _history_path(root: Path, config_path: Path, fold: int) -> Path:
    return _run_dir(root, config_path, fold) / "history.csv"


def _best_validation(path: Path) -> dict[str, float | int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    best = max(rows, key=lambda row: float(row["val_f1_macro"]))
    return {
        "best_epoch": int(best["epoch"]),
        "train_accuracy": float(best["train_accuracy"]),
        "train_f1_macro": float(best["train_f1_macro"]),
        "val_accuracy": float(best["val_accuracy"]),
        "val_f1_macro": float(best["val_f1_macro"]),
        "val_loss": float(best["val_loss"]),
    }


def _write_summary(root: Path, config_paths: list[Path], fold: int) -> tuple[Path, Path]:
    rows: list[dict[str, str | float | int]] = []
    for config_path in config_paths:
        history_path = _history_path(root, config_path, fold)
        if not history_path.exists():
            raise FileNotFoundError(f"Missing completed run history: {history_path}")
        row: dict[str, str | float | int] = {
            "config": str(config_path.relative_to(root)),
            "run_name": _run_name(config_path),
            "fold": fold,
        }
        row.update(_best_validation(history_path))
        rows.append(row)

    rows.sort(key=lambda row: float(row["val_f1_macro"]), reverse=True)
    output_path = root / "results" / f"cnn_augmentation_ablation_fold{fold}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config", "run_name", "fold", *SUMMARY_KEYS])
        writer.writeheader()
        writer.writerows(rows)

    print("\nCNN augmentation ranking (validation Macro F1):")
    for index, row in enumerate(rows, start=1):
        print(
            f"{index}. {row['run_name']}: "
            f"F1={float(row['val_f1_macro']):.4f}, "
            f"accuracy={float(row['val_accuracy']):.4f}, epoch={int(row['best_epoch'])}"
        )
    print(f"Wrote comparison to {output_path}")
    winning_config = root / str(rows[0]["config"])
    return output_path, winning_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and rank CNN spectrogram augmentation profiles.")
    parser.add_argument("--fold", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--evaluate-best",
        action="store_true",
        help="Explicitly evaluate the winning profile on the test fold after validation ranking.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = ROOT
    config_paths = [(root / config).resolve() for config in args.configs]

    if not args.summarize_only:
        for config_path in config_paths:
            history_path = _history_path(root, config_path, args.fold)
            if args.skip_existing and history_path.exists():
                print(f"Skipping completed run: {history_path.parent.name}")
                continue
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.train",
                    "--config",
                    str(config_path),
                    "--fold",
                    str(args.fold),
                ],
                cwd=root,
                check=True,
            )

    _, winning_config = _write_summary(root, config_paths, args.fold)
    winning_run_dir = _run_dir(root, winning_config, args.fold)
    print(f"Selected by validation Macro F1: {winning_run_dir.name}")
    if args.evaluate_best:
        subprocess.run(
            [sys.executable, "-m", "src.evaluate", "--run-dir", str(winning_run_dir)],
            cwd=root,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
