from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


DEFAULT_CONFIGS = [
    "configs/cnn_breakthrough_control.yaml",
    "configs/cnn_breakthrough_delta.yaml",
    "configs/cnn_breakthrough_cooldown.yaml",
    "configs/cnn_breakthrough_single_balance.yaml",
    "configs/cnn_breakthrough_se.yaml",
]
DEFAULT_VALIDATION_FOLDS = (1, 4, 7)


def _write_yaml(config: dict, path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _validation_offset(locked_test_fold: int, validation_fold: int) -> int:
    offset = (validation_fold - locked_test_fold) % 10
    if offset == 0:
        raise ValueError("The validation fold must differ from the locked test fold.")
    return offset


def _candidate_config(base_config: dict, validation_fold: int, locked_test_fold: int) -> dict:
    config = copy.deepcopy(base_config)
    base_name = str(base_config.get("run_name", base_config["model"]["name"]))
    config["run_name"] = f"{base_name}_devval{validation_fold}"
    config.setdefault("data", {})["val_fold_offset"] = _validation_offset(
        locked_test_fold,
        validation_fold,
    )
    config.setdefault("evaluation", {})["run_test"] = False
    return config


def _run_dir(config: dict, locked_test_fold: int) -> Path:
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    return results_dir / f"{config['run_name']}_fold{locked_test_fold}"


def _resolved_config_matches(run_dir: Path, config: dict, locked_test_fold: int) -> bool:
    path = run_dir / "config_resolved.json"
    if not path.exists():
        return False
    resolved = json.loads(path.read_text(encoding="utf-8"))
    return resolved.get("config") == config and int(resolved.get("fold", -1)) == locked_test_fold


def _run_or_reuse(config_path: Path, config: dict, locked_test_fold: int, skip_existing: bool) -> tuple[dict, float]:
    run_dir = _run_dir(config, locked_test_fold)
    metrics_path = run_dir / "validation_metrics.json"
    if metrics_path.exists():
        if not skip_existing:
            raise FileExistsError(f"Run already exists: {run_dir}. Use --skip-existing to resume.")
        if not _resolved_config_matches(run_dir, config, locked_test_fold):
            raise ValueError(f"Existing run uses a different resolved config: {run_dir}")
        return json.loads(metrics_path.read_text(encoding="utf-8")), 0.0

    started = time.monotonic()
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            "--config",
            str(config_path),
            "--fold",
            str(locked_test_fold),
        ],
        cwd=ROOT,
        check=True,
    )
    duration = time.monotonic() - started
    if not metrics_path.exists():
        raise FileNotFoundError(f"Training did not produce validation metrics: {metrics_path}")
    if (run_dir / "metrics.json").exists():
        raise RuntimeError(f"Test metrics were unexpectedly produced during development search: {run_dir}")
    return json.loads(metrics_path.read_text(encoding="utf-8")), duration


def _aggregate(rows: list[dict]) -> list[dict]:
    import statistics

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate"]), []).append(row)

    aggregates = []
    for candidate, candidate_rows in grouped.items():
        f1_values = [float(row["val_f1_macro"]) for row in candidate_rows]
        accuracy_values = [float(row["val_accuracy"]) for row in candidate_rows]
        aggregates.append(
            {
                "candidate": candidate,
                "validation_folds": [int(row["validation_fold"]) for row in candidate_rows],
                "mean_val_f1_macro": statistics.fmean(f1_values),
                "std_val_f1_macro": statistics.pstdev(f1_values),
                "mean_val_accuracy": statistics.fmean(accuracy_values),
                "std_val_accuracy": statistics.pstdev(accuracy_values),
                "total_duration_seconds": sum(float(row["duration_seconds"]) for row in candidate_rows),
            }
        )
    return sorted(aggregates, key=lambda row: float(row["mean_val_f1_macro"]), reverse=True)


def _backup_run(run_dir: Path, config_path: Path, backup_dir: Path | None) -> None:
    if backup_dir is None:
        return
    shutil.copytree(run_dir, backup_dir / "runs" / run_dir.name, dirs_exist_ok=True)
    configs_dir = backup_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, configs_dir / config_path.name)


def _write_progress(rows: list[dict], progress_csv: Path, progress_json: Path, backup_dir: Path | None) -> None:
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    with progress_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    progress_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(progress_csv, backup_dir / progress_csv.name)
        shutil.copy2(progress_json, backup_dir / progress_json.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare high-upside CNN candidates without evaluating the locked fold 10 test set."
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--validation-folds", nargs="+", type=int, default=list(DEFAULT_VALIDATION_FOLDS))
    parser.add_argument("--locked-test-fold", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--search-id", default=time.strftime("%Y%m%dT%H%M%S"))
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation_folds = [int(fold) for fold in args.validation_folds]
    if len(set(validation_folds)) != len(validation_folds):
        raise ValueError("Validation folds must be unique.")
    if any(fold < 1 or fold > 10 for fold in validation_folds):
        raise ValueError("Validation folds must be between 1 and 10.")
    if args.locked_test_fold in validation_folds:
        raise ValueError("The locked test fold cannot be used as a development validation fold.")

    source_paths = [(ROOT / path).resolve() for path in args.configs]
    generated_dir = ROOT / "results" / f"cnn_breakthrough_{args.search_id}_configs"
    summary_csv = ROOT / "results" / f"cnn_breakthrough_{args.search_id}.csv"
    summary_json = ROOT / "results" / f"cnn_breakthrough_{args.search_id}.json"
    progress_csv = ROOT / "results" / f"cnn_breakthrough_{args.search_id}_progress.csv"
    progress_json = ROOT / "results" / f"cnn_breakthrough_{args.search_id}_progress.json"
    backup_dir = args.backup_root / f"cnn_breakthrough_{args.search_id}" if args.backup_root else None

    if args.plan_only:
        print("Candidates:", ", ".join(path.stem for path in source_paths))
        print("Development validation folds:", ", ".join(map(str, validation_folds)))
        print("Locked test fold:", args.locked_test_fold)
        print("Primary metric: mean validation Macro F1")
        print("Test evaluation: disabled")
        return 0

    rows: list[dict] = []
    for source_path in source_paths:
        base_config = load_config(source_path)
        candidate_name = str(base_config.get("run_name", source_path.stem))
        if bool(base_config.get("evaluation", {}).get("run_test", True)):
            raise ValueError(f"Breakthrough config must set evaluation.run_test: false: {source_path}")
        for validation_fold in validation_folds:
            config = _candidate_config(base_config, validation_fold, args.locked_test_fold)
            config_path = generated_dir / f"{config['run_name']}.yaml"
            _write_yaml(config, config_path)
            started_at = datetime.now(timezone.utc)
            metrics, duration = _run_or_reuse(
                config_path,
                config,
                args.locked_test_fold,
                args.skip_existing,
            )
            run_dir = _run_dir(config, args.locked_test_fold)
            row = {
                "candidate": candidate_name,
                "source_config": str(source_path.relative_to(ROOT)),
                "run_name": config["run_name"],
                "locked_test_fold": args.locked_test_fold,
                "validation_fold": validation_fold,
                "best_epoch": int(metrics["best_epoch"]),
                "val_f1_macro": float(metrics["val_f1_macro"]),
                "val_accuracy": float(metrics["val_accuracy"]),
                "val_loss": float(metrics["val_loss"]),
                "duration_seconds": duration,
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            rows.append(row)
            _backup_run(run_dir, config_path, backup_dir)
            _write_progress(rows, progress_csv, progress_json, backup_dir)
            print(
                f"DEVELOPMENT_RESULT candidate={candidate_name} val_fold={validation_fold} "
                f"macro_f1={row['val_f1_macro']:.6f} accuracy={row['val_accuracy']:.6f} "
                f"duration={duration:.1f}s",
                flush=True,
            )

    aggregates = _aggregate(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "selection_metric": "mean validation Macro F1",
        "locked_test_fold": args.locked_test_fold,
        "test_evaluated": False,
        "per_fold": rows,
        "ranking": aggregates,
        "winner": aggregates[0],
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(summary_csv, backup_dir / summary_csv.name)
        shutil.copy2(summary_json, backup_dir / summary_json.name)

    print("\nCNN breakthrough ranking (mean validation Macro F1):")
    for index, row in enumerate(aggregates, start=1):
        print(
            f"{index}. {row['candidate']}: {row['mean_val_f1_macro']:.4f} "
            f"+/- {row['std_val_f1_macro']:.4f}"
        )
    print(f"SEARCH_COMPLETE winner={aggregates[0]['candidate']} summary={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
