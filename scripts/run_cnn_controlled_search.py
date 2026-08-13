from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config


INITIAL_CONFIGS = [
    "configs/cnn_aug_control.yaml",
    "configs/cnn_aug_light.yaml",
    "configs/cnn_aug_balanced.yaml",
    "configs/cnn_aug_strong.yaml",
]
LOG_FIELDS = [
    "sequence",
    "phase",
    "round",
    "run_name",
    "changed_variable",
    "changed_value",
    "config_path",
    "parameters_json",
    "start_time_utc",
    "end_time_utc",
    "duration_seconds",
    "cumulative_seconds",
    "best_epoch",
    "train_macro_f1",
    "validation_macro_f1",
    "validation_accuracy",
    "validation_loss",
    "improved_best",
    "decision",
    "reason",
    "next_adjustment",
    "estimated_remaining_seconds",
    "status",
    "error",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _nested_get(config: dict, path: tuple[str, ...], default=None):
    current = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _nested_set(config: dict, path: tuple[str, ...], value) -> None:
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


@dataclass(frozen=True)
class Mutation:
    name: str
    apply: Callable[[dict], str]


def _set_spec_probability(config: dict) -> str:
    augmentation = config["training"].setdefault("augmentation", {"enabled": True})
    augmentation["enabled"] = True
    spec = augmentation.setdefault("spec_augment", {})
    current = float(spec.get("probability", 0.0))
    value = 0.20 if current == 0.0 else max(0.10, round(current - 0.10, 2))
    spec.update(
        {
            "probability": value,
            "frequency_mask_param": int(spec.get("frequency_mask_param", 8)),
            "time_mask_param": int(spec.get("time_mask_param", 16)),
            "num_frequency_masks": int(spec.get("num_frequency_masks", 1)),
            "num_time_masks": int(spec.get("num_time_masks", 1)),
        }
    )
    return str(value)


def _scale_positive_int(config: dict, path: tuple[str, ...], default: int, factor: float) -> str:
    augmentation = config["training"].setdefault("augmentation", {"enabled": True})
    augmentation["enabled"] = True
    spec = augmentation.setdefault("spec_augment", {"probability": 0.20})
    spec.setdefault("probability", 0.20)
    current = int(_nested_get(config, path, default))
    value = max(1, int(round(current * factor)))
    if value == current:
        value = max(1, current - 1)
    _nested_set(config, path, value)
    return str(value)


def _set_mixup_probability(config: dict) -> str:
    batch_mix = config["training"].setdefault("batch_mix", {})
    current = float(batch_mix.get("probability", 0.0)) if batch_mix.get("enabled", False) else 0.0
    value = 0.10 if current == 0.0 else max(0.05, round(current - 0.10, 2))
    batch_mix.update(
        {
            "enabled": True,
            "mode": "mixup",
            "probability": value,
            "mixup_alpha": float(batch_mix.get("mixup_alpha", 0.20)),
            "cutmix_alpha": float(batch_mix.get("cutmix_alpha", 1.0)),
        }
    )
    return str(value)


def _set_mixup_alpha(config: dict) -> str:
    batch_mix = config["training"].setdefault("batch_mix", {})
    current = float(batch_mix.get("mixup_alpha", 0.20))
    value = 0.10 if current >= 0.20 else 0.20
    batch_mix.update(
        {
            "enabled": True,
            "mode": "mixup",
            "probability": float(batch_mix.get("probability", 0.10)),
            "mixup_alpha": value,
            "cutmix_alpha": float(batch_mix.get("cutmix_alpha", 1.0)),
        }
    )
    return str(value)


def _set_scalar(config: dict, path: tuple[str, ...], value: float) -> str:
    _nested_set(config, path, value)
    return str(value)


MUTATIONS = [
    Mutation("spec_augment_probability", _set_spec_probability),
    Mutation(
        "spec_augment_time_mask",
        lambda config: _scale_positive_int(
            config,
            ("training", "augmentation", "spec_augment", "time_mask_param"),
            default=16,
            factor=0.75,
        ),
    ),
    Mutation(
        "spec_augment_frequency_mask",
        lambda config: _scale_positive_int(
            config,
            ("training", "augmentation", "spec_augment", "frequency_mask_param"),
            default=8,
            factor=0.75,
        ),
    ),
    Mutation("mixup_probability", _set_mixup_probability),
    Mutation("mixup_alpha", _set_mixup_alpha),
    Mutation(
        "class_aware_sampling_power",
        lambda config: _set_scalar(config, ("training", "class_aware_sampling", "power"), 0.35),
    ),
    Mutation(
        "label_smoothing",
        lambda config: _set_scalar(config, ("training", "label_smoothing"), 0.01),
    ),
    Mutation(
        "learning_rate",
        lambda config: _set_scalar(config, ("training", "learning_rate"), 0.00035),
    ),
    Mutation(
        "weight_decay",
        lambda config: _set_scalar(config, ("training", "weight_decay"), 0.002),
    ),
    Mutation(
        "scheduler_min_learning_rate",
        lambda config: _set_scalar(config, ("training", "scheduler", "min_learning_rate"), 0.00003),
    ),
]


def _write_yaml(config: dict, path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install dependencies from requirements.txt.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _run_dir(config: dict, fold: int) -> Path:
    results_dir = Path(config.get("outputs", {}).get("results_dir", "results"))
    if not results_dir.is_absolute():
        results_dir = ROOT / results_dir
    return results_dir / f"{config['run_name']}_fold{fold}"


def _figure_path(config: dict, fold: int, suffix: str) -> Path:
    figures_dir = Path(config.get("outputs", {}).get("figures_dir", "figures"))
    if not figures_dir.is_absolute():
        figures_dir = ROOT / figures_dir
    return figures_dir / f"{config['run_name']}_fold{fold}_{suffix}.png"


def _validation_metrics(run_dir: Path) -> dict[str, float | int]:
    metrics_path = run_dir / "validation_metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    history_path = run_dir / "history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Training history is empty: {history_path}")
    row = max(rows, key=lambda item: float(item["val_f1_macro"]))
    metrics = {
        "best_epoch": int(row["epoch"]),
        "train_accuracy": float(row["train_accuracy"]),
        "train_f1_macro": float(row["train_f1_macro"]),
        "train_loss": float(row["train_loss"]),
        "val_accuracy": float(row["val_accuracy"]),
        "val_f1_macro": float(row["val_f1_macro"]),
        "val_loss": float(row["val_loss"]),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def _config_matches(run_dir: Path, config: dict) -> bool:
    path = run_dir / "config_resolved.json"
    if not path.exists():
        return False
    resolved = json.loads(path.read_text(encoding="utf-8"))
    return resolved.get("config") == config


def _run_training(config_path: Path, config: dict, fold: int, skip_existing: bool) -> tuple[dict, float, str, str]:
    run_dir = _run_dir(config, fold)
    if skip_existing and (run_dir / "history.csv").exists():
        if not _config_matches(run_dir, config):
            raise ValueError(f"Existing run has a different resolved config: {run_dir}")
        metrics = _validation_metrics(run_dir)
        now = _timestamp(_utc_now())
        return metrics, 0.0, now, now

    started = _utc_now()
    command = [sys.executable, "-u", "-m", "src.train", "--config", str(config_path), "--fold", str(fold)]
    last_error = ""
    for attempt in range(1, 3):
        print(f"EXPERIMENT_START run={config['run_name']} attempt={attempt} time={_timestamp(started)}", flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            break
        except subprocess.CalledProcessError as exc:
            last_error = f"exit code {exc.returncode}"
            print(f"EXPERIMENT_RETRY run={config['run_name']} error={last_error}", flush=True)
            if attempt == 2:
                raise RuntimeError(f"Training failed twice for {config['run_name']}: {last_error}") from exc
    ended = _utc_now()
    return _validation_metrics(run_dir), (ended - started).total_seconds(), _timestamp(started), _timestamp(ended)


def _backup_artifacts(
    config: dict,
    config_path: Path,
    fold: int,
    backup_dir: Path | None,
    progress_paths: list[Path],
) -> None:
    if backup_dir is None:
        return

    backup_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir(config, fold)
    shutil.copytree(run_dir, backup_dir / "runs" / run_dir.name, dirs_exist_ok=True)
    configs_dir = backup_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, configs_dir / config_path.name)
    history_figure = _figure_path(config, fold, "training_history")
    if history_figure.exists():
        figures_dir = backup_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(history_figure, figures_dir / history_figure.name)
    for path in progress_paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
    print(f"BACKUP_COMPLETE run={config['run_name']} path={backup_dir}", flush=True)


def _backup_progress(backup_dir: Path | None, progress_paths: list[Path]) -> None:
    if backup_dir is None:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in progress_paths:
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)


def _write_progress(rows: list[dict], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CNN controlled augmentation search",
        "",
        "Primary selection metric: validation Macro F1. Fold 10 test is not used for tuning.",
        "",
        "| Seq | Phase | Round | Run | Changed variable | Val Macro F1 | Val Accuracy | Decision | Duration (s) |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        val_f1 = row["validation_macro_f1"]
        val_accuracy = row["validation_accuracy"]
        lines.append(
            f"| {row['sequence']} | {row['phase']} | {row['round']} | {row['run_name']} | "
            f"{row['changed_variable']} | {val_f1} | {val_accuracy} | {row['decision']} | "
            f"{row['duration_seconds']} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _next_adjustment(index: int, max_rounds: int) -> str:
    next_index = index + 1
    if next_index >= min(max_rounds, len(MUTATIONS)):
        return "lock final configuration"
    return MUTATIONS[next_index].name


def _estimated_remaining(durations: list[float], remaining_runs: int) -> float:
    nonzero = [value for value in durations if value > 0.0]
    if not nonzero:
        return 0.0
    return sum(nonzero) / len(nonzero) * max(remaining_runs, 0)


def _record(
    rows: list[dict],
    *,
    phase: str,
    round_number: int | str,
    config: dict,
    config_path: Path,
    changed_variable: str,
    changed_value: str,
    start_time: str,
    end_time: str,
    duration: float,
    cumulative: float,
    metrics: dict,
    improved: bool,
    decision: str,
    reason: str,
    next_adjustment: str,
    estimated_remaining: float,
    status: str = "completed",
    error: str = "",
) -> dict:
    row = {
        "sequence": len(rows) + 1,
        "phase": phase,
        "round": round_number,
        "run_name": config["run_name"],
        "changed_variable": changed_variable,
        "changed_value": changed_value,
        "config_path": str(config_path.relative_to(ROOT)),
        "parameters_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "duration_seconds": f"{duration:.1f}",
        "cumulative_seconds": f"{cumulative:.1f}",
        "best_epoch": metrics.get("best_epoch", ""),
        "train_macro_f1": f"{float(metrics['train_f1_macro']):.6f}" if metrics else "",
        "validation_macro_f1": f"{float(metrics['val_f1_macro']):.6f}" if metrics else "",
        "validation_accuracy": f"{float(metrics['val_accuracy']):.6f}" if metrics else "",
        "validation_loss": f"{float(metrics['val_loss']):.6f}" if metrics else "",
        "improved_best": str(improved).lower(),
        "decision": decision,
        "reason": reason,
        "next_adjustment": next_adjustment,
        "estimated_remaining_seconds": f"{estimated_remaining:.1f}",
        "status": status,
        "error": error,
    }
    rows.append(row)
    return row


def _write_final_report(
    rows: list[dict],
    best_config: dict,
    best_metrics: dict,
    final_metrics: dict | None,
    output_path: Path,
) -> None:
    iteration_rows = [row for row in rows if row["phase"] == "iteration"]
    effective = [row["changed_variable"] for row in iteration_rows if row["improved_best"] == "true"]
    ineffective = [row["changed_variable"] for row in iteration_rows if row["improved_best"] != "true"]
    gap = float(best_metrics["train_f1_macro"]) - float(best_metrics["val_f1_macro"])
    control_row = next(
        (row for row in rows if row["phase"] == "initial" and row["changed_value"] == "control"),
        None,
    )
    control_gap = None
    if control_row is not None:
        control_gap = float(control_row["train_macro_f1"]) - float(control_row["validation_macro_f1"])
    lines = [
        "# CNN controlled search final report",
        "",
        f"- Selected run: `{best_config['run_name']}`",
        f"- Best validation Macro F1: {float(best_metrics['val_f1_macro']):.4f}",
        f"- Validation Accuracy: {float(best_metrics['val_accuracy']):.4f}",
        f"- Train-validation Macro F1 gap at selected epoch: {gap:.4f}",
        f"- Effective adjustments: {', '.join(effective) if effective else 'none'}",
        f"- Non-improving adjustments: {', '.join(ineffective) if ineffective else 'none'}",
    ]
    if control_gap is not None:
        direction = "reduced" if gap < control_gap else "increased"
        lines.extend(
            [
                f"- Control train-validation Macro F1 gap: {control_gap:.4f}",
                f"- Overfitting-gap change versus control: {gap - control_gap:+.4f} ({direction})",
            ]
        )
    if final_metrics is not None:
        test_f1 = float(final_metrics["f1_macro"])
        lines.extend(
            [
                f"- Final fold 10 test Macro F1: {test_f1:.4f}",
                f"- Final fold 10 test Accuracy: {float(final_metrics['accuracy']):.4f}",
                f"- Difference from historical test Macro F1 0.841: {test_f1 - 0.841:+.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "The experiment used validation Macro F1 as the sole primary selection metric. "
            "The fold 10 test set was evaluated only after a single configuration had been locked.",
            "",
            "A final 10-fold cross-validation run is recommended only if the selected configuration "
            "is sufficiently stable and its validation gain justifies the additional compute.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a validation-only controlled CNN augmentation search.")
    parser.add_argument("--fold", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--max-rounds", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--backup-root", type=Path)
    parser.add_argument("--search-id", default=_utc_now().strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.patience < 1:
        raise ValueError("--patience must be at least 1")

    generated_dir = ROOT / "results" / f"cnn_controlled_search_{args.search_id}_configs"
    progress_csv = ROOT / "results" / f"cnn_controlled_search_{args.search_id}.csv"
    progress_md = ROOT / "results" / f"cnn_controlled_search_{args.search_id}.md"
    final_report = ROOT / "results" / f"cnn_controlled_search_{args.search_id}_final_report.md"
    backup_dir = None
    if args.backup_root is not None:
        backup_dir = args.backup_root / f"cnn_controlled_search_{args.search_id}"

    if args.plan_only:
        print("Initial profiles:", ", ".join(INITIAL_CONFIGS))
        print("Iteration plan:", ", ".join(item.name for item in MUTATIONS[: args.max_rounds]))
        print("Primary metric: validation Macro F1")
        print("Test evaluation:", "once after lock" if args.final_test else "disabled")
        print("Backup directory:", backup_dir or "disabled")
        return 0

    rows: list[dict] = []
    durations: list[float] = []
    cumulative = 0.0
    initial_results: list[tuple[dict, Path, dict]] = []

    for index, relative_path in enumerate(INITIAL_CONFIGS):
        source_config_path = ROOT / relative_path
        config = load_config(source_config_path)
        profile_name = source_config_path.stem.removeprefix("cnn_aug_")
        config["run_name"] = f"cnn_aug_{args.search_id}_initial_{profile_name}"
        config.setdefault("evaluation", {})["run_test"] = False
        config_path = generated_dir / f"{config['run_name']}.yaml"
        _write_yaml(config, config_path)
        attempted_at = _utc_now()
        try:
            metrics, duration, started, ended = _run_training(
                config_path,
                config,
                args.fold,
                args.skip_existing,
            )
        except Exception as exc:
            failed_at = _utc_now()
            duration = (failed_at - attempted_at).total_seconds()
            cumulative += duration
            _record(
                rows,
                phase="initial",
                round_number=index + 1,
                config=config,
                config_path=config_path,
                changed_variable="augmentation_profile",
                changed_value=profile_name,
                start_time=_timestamp(attempted_at),
                end_time=_timestamp(failed_at),
                duration=duration,
                cumulative=cumulative,
                metrics={},
                improved=False,
                decision="stop",
                reason="The experiment failed twice.",
                next_adjustment="none",
                estimated_remaining=0.0,
                status="failed",
                error=str(exc),
            )
            _write_progress(rows, progress_csv, progress_md)
            _backup_progress(backup_dir, [progress_csv, progress_md, config_path])
            raise
        durations.append(duration)
        cumulative += duration
        remaining = len(INITIAL_CONFIGS) - index - 1 + args.max_rounds + (1 if args.final_test else 0)
        _record(
            rows,
            phase="initial",
            round_number=index + 1,
            config=config,
            config_path=config_path,
            changed_variable="augmentation_profile",
            changed_value=profile_name,
            start_time=started,
            end_time=ended,
            duration=duration,
            cumulative=cumulative,
            metrics=metrics,
            improved=False,
            decision="candidate",
            reason="Initial controlled comparison; winner selected after all four complete.",
            next_adjustment=INITIAL_CONFIGS[index + 1] if index + 1 < len(INITIAL_CONFIGS) else MUTATIONS[0].name,
            estimated_remaining=_estimated_remaining(durations, remaining),
        )
        _write_progress(rows, progress_csv, progress_md)
        _backup_artifacts(config, config_path, args.fold, backup_dir, [progress_csv, progress_md])
        initial_results.append((config, config_path, metrics))
        print(
            f"PROGRESS initial={index + 1}/4 run={config['run_name']} "
            f"val_f1={float(metrics['val_f1_macro']):.6f} duration={duration:.1f}s "
            f"eta={_estimated_remaining(durations, remaining):.1f}s",
            flush=True,
        )

    best_config, best_config_path, best_metrics = max(
        initial_results,
        key=lambda item: float(item[2]["val_f1_macro"]),
    )
    best_config = copy.deepcopy(best_config)
    best_score = float(best_metrics["val_f1_macro"])
    print(f"INITIAL_WINNER run={best_config['run_name']} val_f1={best_score:.6f}", flush=True)

    consecutive_no_improvement = 0
    completed_rounds = 0
    for index, mutation in enumerate(MUTATIONS[: args.max_rounds]):
        completed_rounds = index + 1
        candidate = copy.deepcopy(best_config)
        changed_value = mutation.apply(candidate)
        candidate["run_name"] = f"cnn_aug_{args.search_id}_iter{index + 1:02d}_{mutation.name}"
        candidate.setdefault("evaluation", {})["run_test"] = False
        candidate_path = generated_dir / f"{candidate['run_name']}.yaml"
        _write_yaml(candidate, candidate_path)

        attempted_at = _utc_now()
        try:
            metrics, duration, started, ended = _run_training(
                candidate_path,
                candidate,
                args.fold,
                args.skip_existing,
            )
        except Exception as exc:
            failed_at = _utc_now()
            duration = (failed_at - attempted_at).total_seconds()
            cumulative += duration
            _record(
                rows,
                phase="iteration",
                round_number=index + 1,
                config=candidate,
                config_path=candidate_path,
                changed_variable=mutation.name,
                changed_value=changed_value,
                start_time=_timestamp(attempted_at),
                end_time=_timestamp(failed_at),
                duration=duration,
                cumulative=cumulative,
                metrics={},
                improved=False,
                decision="stop",
                reason="The experiment failed twice.",
                next_adjustment="none",
                estimated_remaining=0.0,
                status="failed",
                error=str(exc),
            )
            _write_progress(rows, progress_csv, progress_md)
            _backup_progress(backup_dir, [progress_csv, progress_md, candidate_path])
            raise

        durations.append(duration)
        cumulative += duration
        score = float(metrics["val_f1_macro"])
        improved = score > best_score + 1e-9
        if improved:
            reason = f"Validation Macro F1 improved from {best_score:.6f} to {score:.6f}."
            best_config = copy.deepcopy(candidate)
            best_config_path = candidate_path
            best_metrics = metrics
            best_score = score
            consecutive_no_improvement = 0
            decision = "keep"
        else:
            reason = f"Validation Macro F1 did not exceed current best {best_score:.6f}."
            consecutive_no_improvement += 1
            decision = "revert"

        stop_for_patience = consecutive_no_improvement >= args.patience
        next_item = "lock final configuration" if stop_for_patience else _next_adjustment(index, args.max_rounds)
        remaining_rounds = 0 if stop_for_patience else args.max_rounds - index - 1
        remaining = remaining_rounds + (1 if args.final_test else 0)
        _record(
            rows,
            phase="iteration",
            round_number=index + 1,
            config=candidate,
            config_path=candidate_path,
            changed_variable=mutation.name,
            changed_value=changed_value,
            start_time=started,
            end_time=ended,
            duration=duration,
            cumulative=cumulative,
            metrics=metrics,
            improved=improved,
            decision=decision,
            reason=reason,
            next_adjustment=next_item,
            estimated_remaining=_estimated_remaining(durations, remaining),
        )
        _write_progress(rows, progress_csv, progress_md)
        _backup_artifacts(candidate, candidate_path, args.fold, backup_dir, [progress_csv, progress_md])
        print(
            f"PROGRESS iteration={index + 1}/{args.max_rounds} run={candidate['run_name']} "
            f"variable={mutation.name} value={changed_value} val_f1={score:.6f} "
            f"best={best_score:.6f} decision={decision} duration={duration:.1f}s "
            f"eta={_estimated_remaining(durations, remaining):.1f}s",
            flush=True,
        )
        if stop_for_patience:
            print(f"STOP patience={args.patience} completed_rounds={completed_rounds}", flush=True)
            break

    selected_path = generated_dir / f"cnn_aug_{args.search_id}_selected.yaml"
    _write_yaml(best_config, selected_path)
    if backup_dir is not None:
        (backup_dir / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_path, backup_dir / "configs" / selected_path.name)

    final_metrics = None
    if args.final_test:
        selected_run_dir = _run_dir(best_config, args.fold)
        evaluation_path = selected_run_dir / "evaluation_metrics.json"
        if evaluation_path.exists():
            print(f"FINAL_TEST_SKIPPED existing={evaluation_path}", flush=True)
        else:
            command = [sys.executable, "-u", "-m", "src.evaluate", "--run-dir", str(selected_run_dir)]
            for attempt in range(1, 3):
                print(f"FINAL_TEST_START run={best_config['run_name']} attempt={attempt}", flush=True)
                try:
                    subprocess.run(command, cwd=ROOT, check=True)
                    break
                except subprocess.CalledProcessError:
                    if evaluation_path.exists():
                        break
                    if attempt == 2:
                        raise
                    print(f"FINAL_TEST_RETRY run={best_config['run_name']}", flush=True)
        final_metrics = json.loads(evaluation_path.read_text(encoding="utf-8"))
        _backup_artifacts(best_config, best_config_path, args.fold, backup_dir, [progress_csv, progress_md])
        final_figure = _figure_path(best_config, args.fold, "evaluation_confusion_matrix")
        if backup_dir is not None and final_figure.exists():
            (backup_dir / "figures").mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_figure, backup_dir / "figures" / final_figure.name)

    _write_final_report(rows, best_config, best_metrics, final_metrics, final_report)
    if backup_dir is not None:
        shutil.copy2(final_report, backup_dir / final_report.name)
    print(
        f"SEARCH_COMPLETE selected={best_config['run_name']} val_f1={best_score:.6f} "
        f"rounds={completed_rounds} report={final_report} backup={backup_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
