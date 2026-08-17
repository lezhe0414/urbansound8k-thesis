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

from scripts.run_pretrained_cnn_ensemble_evaluation import run_development_evaluation
from scripts.run_pretrained_cnn_seed_study import run_seed_study, train_linear_seed_runs
from src.utils.config import load_config


DEFAULT_SEEDS = (42, 123, 2026)
DEVELOPMENT_FOLDS = (1, 4, 7)


def _validate_postformal_config(config: dict) -> None:
    folds = tuple(int(value) for value in config["data"].get("development_folds", []))
    if folds != DEVELOPMENT_FOLDS:
        raise ValueError(f"Post-formal study requires development folds {DEVELOPMENT_FOLDS}.")
    if int(config["data"].get("sealed_test_fold", -1)) != 10:
        raise ValueError("Post-formal study requires fold 10 to remain sealed.")
    if bool(config.get("evaluation", {}).get("locked_for_test", False)):
        raise PermissionError("Post-formal exploratory configs must not be locked for test evaluation.")
    if bool(config.get("evaluation", {}).get("formal_cross_validation", False)):
        raise PermissionError("Formal cross-validation is forbidden in the post-formal study.")


def _aggregate_single_seed(run_root: Path, checkpoint_metrics_name: str) -> dict:
    rows: list[dict] = []
    for fold in DEVELOPMENT_FOLDS:
        metrics_path = run_root / f"valfold{fold}" / checkpoint_metrics_name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "validation_fold": fold,
                "accuracy": float(metrics["val_accuracy"]),
                "f1_macro": float(metrics["val_f1_macro"]),
            }
        )
    summary = {"folds": rows}
    for metric in ("accuracy", "f1_macro"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[f"validation_{metric}_mean"] = float(values.mean())
        summary[f"validation_{metric}_std"] = float(values.std(ddof=0))
    return summary


def _summary_row(name: str, family: str, method: str, summary: dict) -> dict:
    return {
        "name": name,
        "loss": family,
        "method": method,
        "validation_f1_macro_mean": float(summary["validation_f1_macro_mean"]),
        "validation_f1_macro_std": float(summary["validation_f1_macro_std"]),
        "validation_accuracy_mean": float(summary["validation_accuracy_mean"]),
        "validation_accuracy_std": float(summary["validation_accuracy_std"]),
    }


def run_postformal_study(
    linear_config: dict,
    ce_config: dict,
    focal_config: dict,
    seeds: list[int],
    output_name: str,
    backup_root: Path | None,
) -> dict:
    if tuple(int(seed) for seed in seeds) != DEFAULT_SEEDS:
        raise ValueError(f"The pre-registered seed set is {DEFAULT_SEEDS}.")
    for config in (linear_config, ce_config, focal_config):
        _validate_postformal_config(config)

    results_dir = Path(ce_config.get("outputs", {}).get("results_dir", "results"))
    summary_dir = results_dir / output_name
    if summary_dir.exists():
        raise FileExistsError(f"Refusing to overwrite post-formal summary: {summary_dir}")

    rows: list[dict] = []
    shared_linear_dirs = train_linear_seed_runs(
        linear_config,
        seeds=seeds,
        output_name=output_name,
        backup_root=backup_root,
    )
    for family, config in (("cross_entropy", ce_config), ("focal_gamma_1.5", focal_config)):
        family_output = f"{output_name}_{family}"
        best_ensemble = run_seed_study(
            linear_config,
            config,
            seeds=seeds,
            output_name=family_output,
            tta_offsets_seconds=[0.0],
            backup_root=backup_root,
            linear_dirs_by_seed=shared_linear_dirs,
        )
        run_roots = [results_dir / f"{family_output}_seed{seed}" for seed in seeds]
        seed42_best = _aggregate_single_seed(run_roots[0], "validation_metrics.json")
        seed42_average = _aggregate_single_seed(
            run_roots[0],
            "checkpoint_averaging_metrics.json",
        )
        averaged_ensemble = run_development_evaluation(
            run_roots,
            folds=list(DEVELOPMENT_FOLDS),
            output_dir=results_dir / f"{family_output}_checkpoint_average_ensemble",
            tta_offsets_seconds=[0.0],
            backup_root=backup_root,
            checkpoint_name="averaged_model.pt",
        )
        rows.extend(
            [
                _summary_row(f"{family}_seed42_best", family, "single_seed_best", seed42_best),
                _summary_row(
                    f"{family}_seed42_checkpoint_average",
                    family,
                    "single_seed_checkpoint_average",
                    seed42_average,
                ),
                _summary_row(
                    f"{family}_three_seed_probability",
                    family,
                    "three_seed_probability_ensemble",
                    best_ensemble,
                ),
                _summary_row(
                    f"{family}_three_seed_checkpoint_average",
                    family,
                    "three_seed_probability_ensemble_of_checkpoint_averages",
                    averaged_ensemble,
                ),
            ]
        )

    summary_dir.mkdir(parents=True)
    winner = max(rows, key=lambda row: float(row["validation_f1_macro_mean"]))
    summary = {
        "run_name": output_name,
        "study_type": "post_formal_exploratory_development_only",
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_test_fold": 10,
        "seeds": seeds,
        "selection_metric": "development_validation_macro_f1_mean",
        "formal_test_results_used_for_selection": False,
        "test_evaluated": False,
        "variants": rows,
        "winner": winner,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if backup_root is not None:
        destination = backup_root / output_name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite Drive summary backup: {destination}")
        shutil.copytree(summary_dir, destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run development-only MN20 seed, checkpoint averaging, and focal-loss studies."
    )
    parser.add_argument("--linear-config", default="configs/pretrained_cnn_mn20_linear.yaml")
    parser.add_argument("--ce-config", default="configs/pretrained_cnn_mn20_postformal_ce.yaml")
    parser.add_argument("--focal-config", default="configs/pretrained_cnn_mn20_postformal_focal.yaml")
    parser.add_argument("--seeds", nargs=3, type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--output-name", default="pretrained_cnn_mn20_postformal_exploration")
    parser.add_argument("--backup-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_postformal_study(
        load_config(args.linear_config),
        load_config(args.ce_config),
        load_config(args.focal_config),
        seeds=[int(seed) for seed in args.seeds],
        output_name=args.output_name,
        backup_root=args.backup_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
