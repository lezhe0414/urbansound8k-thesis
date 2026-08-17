from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import evaluate_run
from src.formal_cross_validation import summarize_fold_predictions, validate_fold_predictions
from src.train import _class_names, train_one_fold
from src.utils.config import load_config
from src.utils.plotting import save_confusion_matrix


LOCKED_CONFIG_PATH = ROOT / "configs" / "cnn_aug_final.yaml"
LOCKED_CONFIG_SHA256 = "6831eedade7a0cb6e7d2e2b98d32bd067bcc1c7fe62568a2059ead4fe68b82e4"
CYCLIC_VALIDATION_FOLDS = {fold: (fold % 10) + 1 for fold in range(1, 11)}
SUMMARY_FILENAMES = (
    "protocol.json",
    "summary.json",
    "fold_metrics.csv",
    "aggregate_predictions.npz",
    "aggregate_confusion_matrix.png",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_fold_mapping() -> dict[str, int]:
    return {str(fold): val_fold for fold, val_fold in CYCLIC_VALIDATION_FOLDS.items()}


def validate_locked_config(config: dict) -> None:
    if _file_sha256(LOCKED_CONFIG_PATH) != LOCKED_CONFIG_SHA256:
        raise ValueError("configs/cnn_aug_final.yaml no longer matches the locked formal configuration.")
    if str(config.get("run_name")) != "cnn_aug_final":
        raise ValueError("The locked run_name must remain cnn_aug_final.")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("Formal cross-validation must use the locked seed 42.")
    if str(config.get("model", {}).get("name")) != "cnn":
        raise ValueError("Formal cross-validation requires the locked from-scratch CNN.")
    if bool(config.get("training", {}).get("ema", {}).get("enabled", True)):
        raise ValueError("EMA must remain disabled for the locked formal CNN.")
    if int(config.get("data", {}).get("val_fold_offset", -1)) != 1:
        raise ValueError("Formal validation folds must use the locked cyclic offset of one.")
    if not bool(config.get("evaluation", {}).get("run_test", False)):
        raise ValueError("The locked source config must explicitly authorize final test evaluation.")


def _formal_config(config: dict, output_name: str, study_root: Path) -> dict:
    resolved = copy.deepcopy(config)
    resolved["run_name"] = f"{output_name}_model"
    resolved.setdefault("evaluation", {})["run_test"] = False
    resolved["evaluation"]["formal_cross_validation"] = True
    resolved["evaluation"]["locked_for_test"] = True
    resolved.setdefault("outputs", {})["results_dir"] = str(study_root / "runs")
    resolved["outputs"]["figures_dir"] = str(study_root / "figures")
    return resolved


def _write_new_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copytree_new(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite backup: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def _backup_summary(study_root: Path, backup_study_root: Path) -> None:
    destination = backup_study_root / "formal_10fold_summary"
    if destination.exists():
        for filename in SUMMARY_FILENAMES:
            source_file = study_root / filename
            destination_file = destination / filename
            if not destination_file.exists() or destination_file.read_bytes() != source_file.read_bytes():
                raise ValueError(f"Existing Drive summary differs from local artifact: {filename}")
        return
    destination.mkdir()
    for filename in SUMMARY_FILENAMES:
        shutil.copy2(study_root / filename, destination / filename)


def _archive_incomplete_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir.with_name(f"{run_dir.name}.incomplete-{timestamp}")
    if destination.exists():
        raise FileExistsError(f"Cannot archive incomplete run because destination exists: {destination}")
    run_dir.rename(destination)


def _load_completed_fold(run_dir: Path, fold: int, val_fold: int, num_classes: int) -> dict:
    completion_path = run_dir / "formal_test_completed.json"
    prediction_path = run_dir / "evaluation_predictions.npz"
    metrics_path = run_dir / "evaluation_metrics.json"
    if not completion_path.exists() or not prediction_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Formal fold {fold} is missing completion artifacts in {run_dir}.")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if not bool(completion.get("test_evaluated")) or int(completion.get("test_fold", -1)) != fold:
        raise ValueError(f"Invalid formal completion manifest for fold {fold}.")
    if int(completion.get("validation_fold", -1)) != val_fold:
        raise ValueError(f"Validation-fold mismatch for completed fold {fold}.")
    with np.load(prediction_path, allow_pickle=False) as payload:
        targets = payload["targets"]
        probabilities = payload["probabilities"]
    validate_fold_predictions(targets, probabilities, num_classes)
    return {
        "test_fold": fold,
        "validation_fold": val_fold,
        "targets": targets,
        "probabilities": probabilities,
    }


def _write_summary_files(
    study_root: Path,
    summary: dict,
    aggregate_targets: np.ndarray,
    aggregate_probabilities: np.ndarray,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    summary_path = study_root / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite formal summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = list(summary["folds"])
    with (study_root / "fold_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_predictions = aggregate_probabilities.argmax(axis=1)
    np.savez_compressed(
        study_root / "aggregate_predictions.npz",
        targets=aggregate_targets,
        probabilities=aggregate_probabilities,
        predictions=aggregate_predictions,
    )
    save_confusion_matrix(
        matrix,
        class_names,
        study_root / "aggregate_confusion_matrix.png",
        title="Locked from-scratch CNN formal 10-fold",
    )


def run_formal_cross_validation(output_name: str, backup_root: Path, resume: bool = False) -> Path:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_name) is None:
        raise ValueError("output_name must be a simple filename-safe identifier.")
    source_config = load_config(LOCKED_CONFIG_PATH)
    validate_locked_config(source_config)
    base_results_dir = Path(source_config.get("outputs", {}).get("results_dir", "results"))
    study_root = base_results_dir / output_name
    protocol = {
        "study_type": "locked formal 10-fold cross-validation",
        "selection_metric": "validation Macro F1 within each training run",
        "test_selection_prohibited": True,
        "config_path": str(LOCKED_CONFIG_PATH.relative_to(ROOT)),
        "config_sha256": LOCKED_CONFIG_SHA256,
        "git_commit": _git_commit(),
        "seed": 42,
        "test_folds": list(range(1, 11)),
        "validation_fold_mapping": _json_fold_mapping(),
        "output_name": output_name,
    }
    protocol_path = study_root / "protocol.json"
    if study_root.exists():
        if not resume:
            raise FileExistsError(f"Formal output already exists; use --resume after inspection: {study_root}")
        if not protocol_path.exists():
            raise FileNotFoundError(f"Cannot resume without protocol: {protocol_path}")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("Refusing to resume because the locked protocol has changed.")
    else:
        study_root.mkdir(parents=True)
        _write_new_json(protocol_path, protocol)

    backup_study_root = backup_root / output_name
    backup_study_root.mkdir(parents=True, exist_ok=True)
    backup_protocol = backup_study_root / "protocol.json"
    if not backup_protocol.exists():
        shutil.copy2(protocol_path, backup_protocol)
    elif backup_protocol.read_bytes() != protocol_path.read_bytes():
        raise ValueError("Drive protocol differs from the local locked protocol.")

    config = _formal_config(source_config, output_name, study_root)
    num_classes = int(config["data"].get("num_classes", 10))
    class_names = _class_names(Path(config["data"]["processed_dir"]))
    payloads: list[dict] = []
    for fold, val_fold in CYCLIC_VALIDATION_FOLDS.items():
        run_dir = Path(config["outputs"]["results_dir"]) / f"{config['run_name']}_fold{fold}"
        completion_path = run_dir / "formal_test_completed.json"
        if completion_path.exists():
            payloads.append(_load_completed_fold(run_dir, fold, val_fold, num_classes))
            backup_run_dir = backup_study_root / "runs" / run_dir.name
            if not backup_run_dir.exists():
                _copytree_new(run_dir, backup_run_dir)
            continue

        test_started_path = run_dir / "formal_test_started.json"
        if test_started_path.exists():
            raise RuntimeError(
                f"Fold {fold} test evaluation started but did not complete. "
                "Refusing an automatic second test evaluation; inspect the run first."
            )

        training_complete = all(
            (run_dir / filename).exists()
            for filename in ("best_model.pt", "validation_metrics.json", "config_resolved.json")
        )
        if not training_complete:
            _archive_incomplete_run(run_dir)
            run_dir = train_one_fold(config, fold)

        _write_new_json(
            test_started_path,
            {
                "started_at": _utc_now(),
                "test_fold": fold,
                "validation_fold": val_fold,
                "test_evaluated": False,
            },
        )
        metrics = evaluate_run(run_dir)
        _write_new_json(
            completion_path,
            {
                "completed_at": _utc_now(),
                "test_fold": fold,
                "validation_fold": val_fold,
                "test_evaluated": True,
                "metrics": metrics,
            },
        )
        payloads.append(_load_completed_fold(run_dir, fold, val_fold, num_classes))
        _copytree_new(run_dir, backup_study_root / "runs" / run_dir.name)

    existing_summary_path = study_root / "summary.json"
    if existing_summary_path.exists():
        existing_summary = json.loads(existing_summary_path.read_text(encoding="utf-8"))
        if existing_summary.get("test_folds_evaluated_once") != list(range(1, 11)):
            raise ValueError("Existing formal summary does not contain exactly ten completed test folds.")
        _backup_summary(study_root, backup_study_root)
        print(json.dumps(existing_summary, indent=2, sort_keys=True))
        return study_root

    summary, aggregate_targets, aggregate_probabilities, matrix = summarize_fold_predictions(payloads, class_names)
    summary.update(
        {
            "run_name": output_name,
            "config_sha256": LOCKED_CONFIG_SHA256,
            "git_commit": protocol["git_commit"],
            "seed": 42,
            "validation_fold_mapping": _json_fold_mapping(),
            "model_selection_used_test_metrics": False,
        }
    )
    _write_summary_files(
        study_root,
        summary,
        aggregate_targets,
        aggregate_probabilities,
        matrix,
        class_names,
    )
    _backup_summary(study_root, backup_study_root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return study_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single locked from-scratch CNN formal 10-fold protocol.")
    parser.add_argument("--output-name", required=True, help="Unique immutable formal-study name.")
    parser.add_argument("--backup-root", required=True, help="Google Drive artifact root.")
    parser.add_argument("--resume", action="store_true", help="Resume only folds with a valid locked protocol.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_formal_cross_validation(
        output_name=str(args.output_name),
        backup_root=Path(args.backup_root),
        resume=bool(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
