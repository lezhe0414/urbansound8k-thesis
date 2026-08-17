from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_pretrained_cnn import evaluate_checkpoint_group


def run_development_evaluation(
    run_roots: list[Path],
    folds: list[int],
    output_dir: Path,
    tta_offsets_seconds: list[float],
    backup_root: Path | None = None,
    checkpoint_name: str = "best_model.pt",
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite development evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    rows: list[dict] = []
    for fold in folds:
        checkpoints = [root / f"valfold{fold}" / checkpoint_name for root in run_roots]
        missing = [path for path in checkpoints if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint(s) for fold {fold}: {missing}")
        metrics = evaluate_checkpoint_group(
            checkpoints,
            split="val",
            test_fold=10,
            val_fold=fold,
            output_dir=output_dir / f"valfold{fold}",
            tta_config={"enabled": len(tta_offsets_seconds) > 1, "offsets_seconds": tta_offsets_seconds},
        )
        rows.append(
            {
                "validation_fold": fold,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            }
        )

    summary = {
        "run_name": output_dir.name,
        "selection_metric": "development_validation_macro_f1_mean",
        "source_run_roots": [str(path) for path in run_roots],
        "checkpoint_count_per_fold": len(run_roots),
        "development_folds": folds,
        "tta_offsets_seconds": tta_offsets_seconds,
        "checkpoint_name": checkpoint_name,
        "folds": rows,
        "test_evaluated": False,
    }
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        summary[f"validation_{metric}_mean"] = float(values.mean())
        summary[f"validation_{metric}_std"] = float(values.std(ddof=0))
    (output_dir / "development_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "development_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if backup_root is not None:
        destination = backup_root / output_dir.name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite Drive evaluation backup: {destination}")
        shutil.copytree(output_dir, destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EfficientAT TTA or seed ensembles on development folds.")
    parser.add_argument("--run-roots", nargs="+", required=True, type=Path)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 4, 7])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tta-offsets-seconds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_development_evaluation(
        args.run_roots,
        folds=[int(fold) for fold in args.folds],
        output_dir=args.output_dir,
        tta_offsets_seconds=[float(value) for value in args.tta_offsets_seconds],
        backup_root=args.backup_root,
        checkpoint_name=args.checkpoint_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
