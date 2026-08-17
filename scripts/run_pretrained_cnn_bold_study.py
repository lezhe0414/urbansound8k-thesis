from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pretrained_cnn_ensemble_evaluation import run_development_evaluation
from src.train_pretrained_cnn import train_validation_fold
from src.utils.config import load_config


DEVELOPMENT_FOLDS = (1, 4, 7)
SCREEN_SEED = 42
EXPANSION_SEEDS = (42, 123, 2026)
LOCKED_SINGLE_SEED_CONTROL_F1 = 0.890685550
LOCKED_THREE_SEED_FOCAL_F1 = 0.893950613

CANDIDATE_CONFIGS = {
    "mn20_control": "configs/pretrained_cnn_bold_mn20_control.yaml",
    "mn20_bnfreeze": "configs/pretrained_cnn_bold_mn20_bnfreeze.yaml",
    "mn30": "configs/pretrained_cnn_bold_mn30.yaml",
    "mn40": "configs/pretrained_cnn_bold_mn40.yaml",
}
ENSEMBLE_MEMBERS = {
    "mn20_mn30": ("mn20_control", "mn30"),
    "mn20_mn40": ("mn20_control", "mn40"),
    "mn30_mn40": ("mn30", "mn40"),
    "mn20_mn30_mn40": ("mn20_control", "mn30", "mn40"),
    "bnfreeze_mn30": ("mn20_bnfreeze", "mn30"),
    "bnfreeze_mn40": ("mn20_bnfreeze", "mn40"),
    "bnfreeze_mn30_mn40": ("mn20_bnfreeze", "mn30", "mn40"),
}


def _validate_config(config: dict) -> None:
    folds = tuple(int(value) for value in config["data"].get("development_folds", []))
    if folds != DEVELOPMENT_FOLDS:
        raise ValueError(f"Bold study requires development folds {DEVELOPMENT_FOLDS}.")
    if int(config["data"].get("sealed_test_fold", -1)) != 10:
        raise ValueError("Bold study requires fold 10 to remain sealed.")
    evaluation = config.get("evaluation", {})
    if bool(evaluation.get("locked_for_test", False)):
        raise PermissionError("Bold exploratory configs must not be locked for test evaluation.")
    if bool(evaluation.get("formal_cross_validation", False)):
        raise PermissionError("Formal cross-validation is forbidden in the bold study.")
    if config["model"].get("stage") != "partial_finetune":
        raise ValueError("Bold candidate configs must use partial_finetune.")


def _linear_config(candidate: dict, output_name: str, seed: int) -> dict:
    config = copy.deepcopy(candidate)
    variant = str(config["model"]["variant"])
    config["run_name"] = f"{output_name}_{variant}_linear_seed{seed}"
    config["seed"] = int(seed)
    config["model"]["stage"] = "linear_probe"
    config["model"]["freeze_encoder_batchnorm"] = False
    training = config["training"]
    training["batch_size"] = 32
    training["epochs"] = 5
    training.pop("initial_checkpoint_template", None)
    training["encoder_learning_rate"] = 0.00001
    training["loss"] = {"name": "cross_entropy"}
    training["gradual_unfreezing"] = {"enabled": False, "head_only_epochs": 0}
    training["mixup"] = {"enabled": False, "probability": 0.0, "alpha": 0.15}
    training["checkpoint_averaging"] = {"enabled": False}
    return config


def _run_root(config: dict) -> Path:
    return Path(config.get("outputs", {}).get("results_dir", "results")) / str(config["run_name"])


def _train_linear(candidate: dict, output_name: str, seed: int, backup_root: Path | None) -> dict[int, Path]:
    config = _linear_config(candidate, output_name, seed)
    paths: dict[int, Path] = {}
    for fold in DEVELOPMENT_FOLDS:
        paths[fold] = train_validation_fold(config, val_fold=fold, backup_root=backup_root)
    return paths


def _train_candidate(
    candidate: dict,
    candidate_name: str,
    output_name: str,
    seed: int,
    linear_paths: dict[int, Path],
    backup_root: Path | None,
) -> Path:
    config = copy.deepcopy(candidate)
    config["seed"] = int(seed)
    config["run_name"] = f"{output_name}_{candidate_name}_seed{seed}"
    for fold in DEVELOPMENT_FOLDS:
        fold_config = copy.deepcopy(config)
        fold_config["training"]["initial_checkpoint_template"] = str(
            linear_paths[fold] / "best_model.pt"
        )
        train_validation_fold(fold_config, val_fold=fold, backup_root=backup_root)
    return _run_root(config)


def _row(name: str, method: str, members: tuple[str, ...], summary: dict) -> dict:
    return {
        "name": name,
        "method": method,
        "members": "+".join(members),
        "validation_f1_macro_mean": float(summary["validation_f1_macro_mean"]),
        "validation_f1_macro_std": float(summary["validation_f1_macro_std"]),
        "validation_accuracy_mean": float(summary["validation_accuracy_mean"]),
        "validation_accuracy_std": float(summary["validation_accuracy_std"]),
    }


def _evaluate_roots(
    roots: list[Path],
    output_dir: Path,
    backup_root: Path | None,
) -> dict:
    return run_development_evaluation(
        roots,
        folds=list(DEVELOPMENT_FOLDS),
        output_dir=output_dir,
        tta_offsets_seconds=[0.0],
        backup_root=backup_root,
    )


def _write_summary(summary_dir: Path, payload: dict, rows: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_bold_study(
    configs: dict[str, dict],
    output_name: str,
    backup_root: Path | None,
    screen_only: bool = False,
) -> dict:
    for config in configs.values():
        _validate_config(config)
    results_dir = Path(next(iter(configs.values())).get("outputs", {}).get("results_dir", "results"))
    summary_dir = results_dir / output_name
    if summary_dir.exists():
        raise FileExistsError(f"Refusing to overwrite bold-study summary: {summary_dir}")

    linear_by_variant: dict[str, dict[int, Path]] = {}
    roots: dict[str, Path] = {}
    for name, config in configs.items():
        variant = str(config["model"]["variant"])
        if variant not in linear_by_variant:
            linear_by_variant[variant] = _train_linear(
                config,
                output_name=output_name,
                seed=SCREEN_SEED,
                backup_root=backup_root,
            )
        roots[name] = _train_candidate(
            config,
            candidate_name=name,
            output_name=output_name,
            seed=SCREEN_SEED,
            linear_paths=linear_by_variant[variant],
            backup_root=backup_root,
        )

    rows: list[dict] = []
    for name, root in roots.items():
        result = _evaluate_roots(
            [root],
            output_dir=results_dir / f"{output_name}_{name}_seed42_summary",
            backup_root=backup_root,
        )
        rows.append(_row(name, "single_model_seed42", (name,), result))
    for name, members in ENSEMBLE_MEMBERS.items():
        result = _evaluate_roots(
            [roots[member] for member in members],
            output_dir=results_dir / f"{output_name}_{name}_cross_scale",
            backup_root=backup_root,
        )
        rows.append(_row(name, "cross_scale_probability_ensemble", members, result))

    screen_winner = max(rows, key=lambda row: float(row["validation_f1_macro_mean"]))
    new_individuals = [row for row in rows if row["name"] != "mn20_control" and row["method"] == "single_model_seed42"]
    expansion_candidate = max(new_individuals, key=lambda row: float(row["validation_f1_macro_mean"]))
    expansion_performed = False
    if not screen_only and float(expansion_candidate["validation_f1_macro_mean"]) >= LOCKED_SINGLE_SEED_CONTROL_F1:
        candidate_name = str(expansion_candidate["name"])
        candidate = configs[candidate_name]
        expanded_roots = [roots[candidate_name]]
        for seed in EXPANSION_SEEDS[1:]:
            linear = _train_linear(candidate, output_name=output_name, seed=seed, backup_root=backup_root)
            expanded_roots.append(
                _train_candidate(
                    candidate,
                    candidate_name=candidate_name,
                    output_name=output_name,
                    seed=seed,
                    linear_paths=linear,
                    backup_root=backup_root,
                )
            )
        result = _evaluate_roots(
            expanded_roots,
            output_dir=results_dir / f"{output_name}_{candidate_name}_3seed_ensemble",
            backup_root=backup_root,
        )
        rows.append(
            _row(
                f"{candidate_name}_3seed",
                "three_seed_probability_ensemble",
                tuple(f"{candidate_name}_seed{seed}" for seed in EXPANSION_SEEDS),
                result,
            )
        )
        expansion_performed = True

    winner = max(rows, key=lambda row: float(row["validation_f1_macro_mean"]))
    payload = {
        "run_name": output_name,
        "study_type": "bold_breakthrough_development_only",
        "development_folds": list(DEVELOPMENT_FOLDS),
        "sealed_test_fold": 10,
        "selection_metric": "development_validation_macro_f1_mean",
        "formal_test_results_used_for_selection": False,
        "test_evaluated": False,
        "screen_seed": SCREEN_SEED,
        "expansion_seeds": list(EXPANSION_SEEDS),
        "expansion_threshold": LOCKED_SINGLE_SEED_CONTROL_F1,
        "historical_three_seed_reference": LOCKED_THREE_SEED_FOCAL_F1,
        "screen_winner": screen_winner,
        "expansion_candidate": expansion_candidate,
        "expansion_performed": expansion_performed,
        "variants": rows,
        "winner": winner,
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
        description="Run development-only EfficientAT scale, BatchNorm, and cross-scale ensemble experiments."
    )
    parser.add_argument("--output-name", default="pretrained_cnn_bold_breakthrough")
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--screen-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_bold_study(
        {name: load_config(path) for name, path in CANDIDATE_CONFIGS.items()},
        output_name=args.output_name,
        backup_root=args.backup_root,
        screen_only=args.screen_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
