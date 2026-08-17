from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pretrained_cnn_bold_study import (
    DEVELOPMENT_FOLDS,
    EXPANSION_SEEDS,
    LOCKED_THREE_SEED_FOCAL_F1,
    _evaluate_roots,
    _linear_config,
    _row,
    _run_root,
    _train_candidate,
    _train_linear,
    _validate_config,
    _write_summary,
)
from src.utils.config import load_config


MN20_CONFIG = "configs/pretrained_cnn_bold_mn20_control.yaml"
MN40_CONFIG = "configs/pretrained_cnn_bold_mn40.yaml"
LOCKED_FIXED_SEED_CROSS_SCALE_F1 = 0.9012805520832051


def _validate_checkpoint_root(root: Path, label: str) -> Path:
    missing = [
        root / f"valfold{fold}" / "best_model.pt"
        for fold in DEVELOPMENT_FOLDS
        if not (root / f"valfold{fold}" / "best_model.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete {label} checkpoint root {root}: {missing}")
    return root


def _load_base_summary(results_dir: Path, base_output_name: str) -> dict:
    path = results_dir / base_output_name / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing source bold-study summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("test_evaluated") is not False:
        raise PermissionError("Source bold study must have test_evaluated=false.")
    if payload.get("formal_test_results_used_for_selection") is not False:
        raise PermissionError("Source bold study must not use formal test results for selection.")
    if tuple(payload.get("development_folds", [])) != DEVELOPMENT_FOLDS:
        raise ValueError(f"Source bold study must use development folds {DEVELOPMENT_FOLDS}.")
    if int(payload.get("sealed_test_fold", -1)) != 10:
        raise PermissionError("Source bold study must keep fold 10 sealed.")
    names = {str(row.get("name")) for row in payload.get("variants", [])}
    if "mn20_mn40" not in names:
        raise ValueError("Source bold study does not contain the preregistered MN20 + MN40 screen.")
    return payload


def _existing_model_root(
    results_dir: Path,
    base_output_name: str,
    candidate_name: str,
    seed: int,
) -> Path:
    return _validate_checkpoint_root(
        results_dir / f"{base_output_name}_{candidate_name}_seed{seed}",
        f"{candidate_name} seed {seed}",
    )


def _ensure_mn20_root(
    config: dict,
    results_dir: Path,
    base_output_name: str,
    seed: int,
    backup_root: Path | None,
) -> tuple[Path, bool]:
    root = results_dir / f"{base_output_name}_mn20_control_seed{seed}"
    if root.exists():
        return _validate_checkpoint_root(root, f"mn20_control seed {seed}"), False

    linear_config = _linear_config(config, base_output_name, seed)
    linear_root = _run_root(linear_config)
    if linear_root.exists():
        _validate_checkpoint_root(linear_root, f"mn20 linear seed {seed}")
        linear_paths = {fold: linear_root / f"valfold{fold}" for fold in DEVELOPMENT_FOLDS}
    else:
        linear_paths = _train_linear(
            config,
            output_name=base_output_name,
            seed=seed,
            backup_root=backup_root,
        )
    trained_root = _train_candidate(
        config,
        candidate_name="mn20_control",
        output_name=base_output_name,
        seed=seed,
        linear_paths=linear_paths,
        backup_root=backup_root,
    )
    return _validate_checkpoint_root(trained_root, f"mn20_control seed {seed}"), True


def run_multiseed_cross_scale(
    mn20_config: dict,
    mn40_config: dict,
    base_output_name: str,
    output_name: str,
    backup_root: Path | None,
) -> dict:
    for config in (mn20_config, mn40_config):
        _validate_config(config)
    if mn20_config["model"]["variant"] != "mn20_as":
        raise ValueError("MN20 config must use the mn20_as variant.")
    if mn40_config["model"]["variant"] != "mn40_as":
        raise ValueError("MN40 config must use the mn40_as variant.")

    results_dir = Path(mn20_config.get("outputs", {}).get("results_dir", "results"))
    if results_dir != Path(mn40_config.get("outputs", {}).get("results_dir", "results")):
        raise ValueError("MN20 and MN40 configs must use the same results directory.")
    summary_dir = results_dir / output_name
    if summary_dir.exists():
        raise FileExistsError(f"Refusing to overwrite multiseed cross-scale summary: {summary_dir}")

    source_summary = _load_base_summary(results_dir, base_output_name)
    mn20_roots: dict[int, Path] = {}
    mn40_roots: dict[int, Path] = {}
    newly_trained_mn20_seeds: list[int] = []
    for seed in EXPANSION_SEEDS:
        mn20_roots[seed], trained = _ensure_mn20_root(
            mn20_config,
            results_dir=results_dir,
            base_output_name=base_output_name,
            seed=seed,
            backup_root=backup_root,
        )
        if trained:
            newly_trained_mn20_seeds.append(seed)
        mn40_roots[seed] = _existing_model_root(
            results_dir,
            base_output_name=base_output_name,
            candidate_name="mn40",
            seed=seed,
        )

    rows: list[dict] = []
    pairwise_results: dict[int, dict] = {}
    for seed in EXPANSION_SEEDS:
        pairwise_results[seed] = _evaluate_roots(
            [mn20_roots[seed], mn40_roots[seed]],
            output_dir=results_dir / f"{output_name}_mn20_mn40_seed{seed}",
            backup_root=backup_root,
        )
        rows.append(
            _row(
                f"mn20_mn40_seed{seed}",
                "same_seed_cross_scale_probability_ensemble",
                (f"mn20_seed{seed}", f"mn40_seed{seed}"),
                pairwise_results[seed],
            )
        )

    six_roots = [
        root
        for seed in EXPANSION_SEEDS
        for root in (mn20_roots[seed], mn40_roots[seed])
    ]
    if len(six_roots) != 6 or len({str(root) for root in six_roots}) != 6:
        raise RuntimeError("The final cross-scale ensemble must contain six unique checkpoint roots.")
    six_model_result = _evaluate_roots(
        six_roots,
        output_dir=results_dir / f"{output_name}_six_model_ensemble",
        backup_root=backup_root,
    )
    six_model_row = _row(
        "mn20_mn40_3seed_each",
        "six_model_cross_scale_probability_ensemble",
        tuple(
            f"{variant}_seed{seed}"
            for seed in EXPANSION_SEEDS
            for variant in ("mn20", "mn40")
        ),
        six_model_result,
    )
    rows.append(six_model_row)

    primary_f1 = float(six_model_row["validation_f1_macro_mean"])
    payload = {
        "run_name": output_name,
        "study_type": "postformal_multiseed_cross_scale_development_only",
        "source_bold_run": base_output_name,
        "source_bold_commit": source_summary.get("git_commit"),
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_test_fold": 10,
        "selection_metric": "development_validation_macro_f1_mean",
        "formal_test_results_used_for_selection": False,
        "test_evaluated": False,
        "variants": ["mn20_as", "mn40_as"],
        "seeds": list(EXPANSION_SEEDS),
        "checkpoint_count_per_fold": 6,
        "newly_trained_mn20_seeds": newly_trained_mn20_seeds,
        "historical_three_seed_mn20_reference_f1": LOCKED_THREE_SEED_FOCAL_F1,
        "fixed_seed_cross_scale_screen_f1": LOCKED_FIXED_SEED_CROSS_SCALE_F1,
        "primary_result": six_model_row,
        "supports_cross_scale_robustness": primary_f1 > LOCKED_THREE_SEED_FOCAL_F1,
        "replicates_fixed_seed_screen": primary_f1 >= LOCKED_FIXED_SEED_CROSS_SCALE_F1,
        "delta_vs_historical_three_seed_mn20": primary_f1 - LOCKED_THREE_SEED_FOCAL_F1,
        "delta_vs_fixed_seed_cross_scale_screen": primary_f1 - LOCKED_FIXED_SEED_CROSS_SCALE_F1,
        "diagnostic_same_seed_pairs": rows[:-1],
        "test_policy": "Fold 10 is sealed; this runner has no test-evaluation path.",
    }
    _write_summary(summary_dir, payload, rows)
    if backup_root is not None:
        destination = backup_root / output_name
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite Drive summary backup: {destination}")
        shutil.copytree(summary_dir, destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Complete a development-only MN20 + MN40 three-seed cross-scale ensemble "
            "without exposing fold 10."
        )
    )
    parser.add_argument("--base-output-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--backup-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_multiseed_cross_scale(
        load_config(MN20_CONFIG),
        load_config(MN40_CONFIG),
        base_output_name=args.base_output_name,
        output_name=args.output_name,
        backup_root=args.backup_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
