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
from src.stacking import DEFAULT_C_GRID, nested_leave_one_fold_out_stacking
from src.utils.metrics import classification_metrics, confusion_matrix_array
from src.utils.plotting import save_confusion_matrix


DEVELOPMENT_FOLDS = (1, 4, 7)
SEEDS = (42, 123, 2026)
SEALED_TEST_FOLD = 10
MEMBERS = tuple(
    (variant, seed)
    for seed in SEEDS
    for variant in ("mn20", "mn40")
)


def _load_source_summary(results_dir: Path, source_summary_name: str) -> dict:
    path = results_dir / source_summary_name / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source cross-scale summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise PermissionError("Source cross-scale study must have test_evaluated=false.")
    if payload.get("formal_test_results_used_for_selection") is not False:
        raise PermissionError("Source study must not use formal test results for selection.")
    if tuple(payload.get("development_folds", [])) != DEVELOPMENT_FOLDS:
        raise ValueError(f"Source study must use development folds {DEVELOPMENT_FOLDS}.")
    if int(payload.get("sealed_test_fold", -1)) != SEALED_TEST_FOLD:
        raise PermissionError(f"Source study must seal fold {SEALED_TEST_FOLD}.")
    if int(payload.get("checkpoint_count_per_fold", -1)) != len(MEMBERS):
        raise ValueError("Source cross-scale study must contain six checkpoints per fold.")
    return payload


def _member_root(
    results_dir: Path,
    base_output_name: str,
    variant: str,
    seed: int,
) -> Path:
    candidate = "mn20_control" if variant == "mn20" else "mn40"
    root = results_dir / f"{base_output_name}_{candidate}_seed{seed}"
    missing = [
        root / f"valfold{fold}" / "best_model.pt"
        for fold in DEVELOPMENT_FOLDS
        if not (root / f"valfold{fold}" / "best_model.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete source checkpoint root {root}: {missing}")
    return root


def _summary_statistics(rows: list[dict], prefix: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        values = np.asarray(
            [float(row[f"{prefix}_{metric}"]) for row in rows],
            dtype=np.float64,
        )
        summary[f"validation_{metric}_mean"] = float(values.mean())
        summary[f"validation_{metric}_std"] = float(values.std(ddof=0))
    return summary


def _collect_member_predictions(
    roots: dict[str, Path],
    output_dir: Path,
) -> dict[int, dict[str, object]]:
    payloads: dict[int, dict[str, object]] = {}
    for fold in DEVELOPMENT_FOLDS:
        expected_targets: np.ndarray | None = None
        probability_sets: list[np.ndarray] = []
        for member_name, root in roots.items():
            member_output = output_dir / "member_predictions" / member_name / f"valfold{fold}"
            evaluate_checkpoint_group(
                [root / f"valfold{fold}" / "best_model.pt"],
                split="val",
                test_fold=SEALED_TEST_FOLD,
                val_fold=fold,
                output_dir=member_output,
                tta_config={"enabled": False, "offsets_seconds": [0.0]},
            )
            with np.load(member_output / "predictions.npz", allow_pickle=False) as prediction_file:
                targets = prediction_file["targets"].astype(np.int64, copy=True)
                probabilities = prediction_file["probabilities"].astype(np.float64, copy=True)
            if expected_targets is None:
                expected_targets = targets
            elif not np.array_equal(expected_targets, targets):
                raise ValueError(f"Member predictions are misaligned for validation fold {fold}.")
            probability_sets.append(probabilities)
        assert expected_targets is not None
        payloads[fold] = {
            "targets": expected_targets,
            "probability_sets": probability_sets,
        }
    return payloads


def run_stacking_study(
    base_output_name: str,
    source_summary_name: str,
    output_name: str,
    backup_root: Path | None,
) -> dict:
    results_dir = Path("results")
    output_dir = results_dir / output_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite stacking study: {output_dir}")
    source_summary = _load_source_summary(results_dir, source_summary_name)
    roots = {
        f"{variant}_seed{seed}": _member_root(
            results_dir,
            base_output_name,
            variant,
            seed,
        )
        for variant, seed in MEMBERS
    }
    output_dir.mkdir(parents=True)
    fold_payloads = _collect_member_predictions(roots, output_dir)
    nested_results = nested_leave_one_fold_out_stacking(
        fold_payloads,
        labels=list(range(10)),
        c_grid=DEFAULT_C_GRID,
        random_state=42,
    )

    rows: list[dict] = []
    for fold in DEVELOPMENT_FOLDS:
        result = nested_results[fold]
        targets = np.asarray(result["targets"], dtype=np.int64)
        stacked_probabilities = np.asarray(result["stacked_probabilities"], dtype=np.float64)
        baseline_probabilities = np.asarray(result["baseline_probabilities"], dtype=np.float64)
        stacked_predictions = stacked_probabilities.argmax(axis=1)
        baseline_predictions = baseline_probabilities.argmax(axis=1)
        stacked_metrics = classification_metrics(
            targets.tolist(),
            stacked_predictions.tolist(),
            list(range(10)),
        )
        baseline_metrics = classification_metrics(
            targets.tolist(),
            baseline_predictions.tolist(),
            list(range(10)),
        )
        row = {
            "validation_fold": fold,
            "selected_c": float(result["selected_c"]),
            **{f"stacking_{key}": value for key, value in stacked_metrics.items()},
            **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
        }
        rows.append(row)
        fold_dir = output_dir / f"valfold{fold}"
        fold_dir.mkdir()
        np.savez_compressed(
            fold_dir / "predictions.npz",
            targets=targets,
            stacked_probabilities=stacked_probabilities,
            stacked_predictions=stacked_predictions,
            baseline_probabilities=baseline_probabilities,
            baseline_predictions=baseline_predictions,
        )
        fold_payload = {
            "validation_fold": fold,
            "meta_training_folds": result["training_folds"],
            "selected_c": result["selected_c"],
            "inner_scores": result["inner_scores"],
            "stacking_metrics": stacked_metrics,
            "baseline_metrics": baseline_metrics,
            "model_parameters": result["model_parameters"],
            "test_evaluated": False,
        }
        (fold_dir / "metrics_and_model.json").write_text(
            json.dumps(fold_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        save_confusion_matrix(
            confusion_matrix_array(targets.tolist(), stacked_predictions.tolist(), list(range(10))),
            [str(label) for label in range(10)],
            fold_dir / "stacking_confusion_matrix.png",
            title=f"Nested stacking validation fold {fold}",
        )
        save_confusion_matrix(
            confusion_matrix_array(targets.tolist(), baseline_predictions.tolist(), list(range(10))),
            [str(label) for label in range(10)],
            fold_dir / "baseline_confusion_matrix.png",
            title=f"Equal-average validation fold {fold}",
        )

    stacking_summary = _summary_statistics(rows, "stacking")
    baseline_summary = _summary_statistics(rows, "baseline")
    source_f1 = float(source_summary["primary_result"]["validation_f1_macro_mean"])
    reproduced_f1 = float(baseline_summary["validation_f1_macro_mean"])
    if abs(reproduced_f1 - source_f1) > 1e-4:
        raise RuntimeError(
            "Recomputed equal-average baseline does not reproduce the source summary: "
            f"{reproduced_f1} vs {source_f1}."
        )
    delta = (
        float(stacking_summary["validation_f1_macro_mean"])
        - float(baseline_summary["validation_f1_macro_mean"])
    )
    summary = {
        "run_name": output_name,
        "study_type": "postformal_nested_stacking_development_only",
        "source_cross_scale_run": source_summary_name,
        "source_cross_scale_f1": source_f1,
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_test_fold": SEALED_TEST_FOLD,
        "selection_metric": "development_validation_macro_f1_mean",
        "formal_test_results_used_for_selection": False,
        "test_evaluated": False,
        "members": list(roots),
        "checkpoint_count_per_fold": len(roots),
        "outer_protocol": "leave_one_development_fold_out",
        "inner_protocol": "reciprocal_validation_within_outer_training_folds",
        "feature_transform": "standardized_concatenated_clipped_log_probabilities",
        "class_weight": "balanced",
        "c_grid": list(DEFAULT_C_GRID),
        "random_state": 42,
        "folds": rows,
        "stacking": stacking_summary,
        "equal_average_baseline": baseline_summary,
        "delta_f1_macro": delta,
        "improves_equal_average": delta > 0.0,
        "test_policy": "Fold 10 is sealed; this runner has no test-evaluation path.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if backup_root is not None:
        destination = backup_root / output_name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite Drive stacking backup: {destination}")
        shutil.copytree(output_dir, destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run development-only nested logistic stacking for fixed MN20/MN40 members."
    )
    parser.add_argument("--base-output-name", required=True)
    parser.add_argument("--source-summary-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--backup-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_stacking_study(
        base_output_name=args.base_output_name,
        source_summary_name=args.source_summary_name,
        output_name=args.output_name,
        backup_root=args.backup_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
