from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pretrained_cnn_ensemble_evaluation import run_development_evaluation
from src.train_pretrained_cnn import train_validation_fold
from src.utils.config import load_config


def train_linear_seed_runs(
    linear_config: dict,
    seeds: list[int],
    output_name: str,
    backup_root: Path | None = None,
) -> dict[int, dict[int, Path]]:
    folds = [int(value) for value in linear_config["data"].get("development_folds", [1, 4, 7])]
    linear_dirs_by_seed: dict[int, dict[int, Path]] = {}
    for seed in seeds:
        linear = copy.deepcopy(linear_config)
        linear["seed"] = int(seed)
        linear["run_name"] = f"{output_name}_linear_seed{seed}"
        linear_dirs_by_seed[int(seed)] = {}
        for fold in folds:
            linear_dirs_by_seed[int(seed)][fold] = train_validation_fold(
                linear,
                val_fold=fold,
                backup_root=backup_root,
                evaluate_test=False,
            )
    return linear_dirs_by_seed


def run_seed_study(
    linear_config: dict,
    finetune_config: dict,
    seeds: list[int],
    output_name: str,
    tta_offsets_seconds: list[float],
    backup_root: Path | None = None,
    linear_dirs_by_seed: dict[int, dict[int, Path]] | None = None,
) -> dict:
    folds = [int(value) for value in finetune_config["data"].get("development_folds", [1, 4, 7])]
    results_dir = Path(finetune_config.get("outputs", {}).get("results_dir", "results"))
    run_roots: list[Path] = []
    if linear_dirs_by_seed is None:
        linear_dirs_by_seed = train_linear_seed_runs(
            linear_config,
            seeds=seeds,
            output_name=output_name,
            backup_root=backup_root,
        )

    for seed in seeds:
        linear_dirs = linear_dirs_by_seed.get(int(seed), {})
        missing_folds = [fold for fold in folds if fold not in linear_dirs]
        if missing_folds:
            raise FileNotFoundError(
                f"Shared linear checkpoints for seed {seed} are missing folds {missing_folds}."
            )
        finetune = copy.deepcopy(finetune_config)
        finetune["seed"] = int(seed)
        finetune["run_name"] = f"{output_name}_seed{seed}"
        for fold in folds:
            fold_config = copy.deepcopy(finetune)
            fold_config["training"]["initial_checkpoint_template"] = str(
                linear_dirs[fold] / "best_model.pt"
            )
            train_validation_fold(
                fold_config,
                val_fold=fold,
                backup_root=backup_root,
                evaluate_test=False,
            )
        run_roots.append(results_dir / finetune["run_name"])

    return run_development_evaluation(
        run_roots,
        folds=folds,
        output_dir=results_dir / f"{output_name}_ensemble",
        tta_offsets_seconds=tta_offsets_seconds,
        backup_root=backup_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a fixed EfficientAT multi-seed development study.")
    parser.add_argument("--linear-config", required=True)
    parser.add_argument("--finetune-config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--tta-offsets-seconds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--backup-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_seed_study(
        load_config(args.linear_config),
        load_config(args.finetune_config),
        seeds=[int(seed) for seed in args.seeds],
        output_name=args.output_name,
        tta_offsets_seconds=[float(value) for value in args.tta_offsets_seconds],
        backup_root=args.backup_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
