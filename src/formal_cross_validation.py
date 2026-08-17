from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.metrics import classification_metrics, confusion_matrix_array


METRIC_NAMES = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def validate_fold_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    num_classes: int,
) -> None:
    if targets.ndim != 1:
        raise ValueError("Targets must be one-dimensional.")
    if probabilities.ndim != 2 or probabilities.shape[1] != num_classes:
        raise ValueError(f"Probabilities must have shape (examples, {num_classes}).")
    if probabilities.shape[0] != targets.shape[0]:
        raise ValueError("Targets and probabilities must contain the same number of examples.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("Probabilities cannot be negative.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Each probability row must sum to one.")


def summarize_fold_predictions(
    fold_payloads: list[dict[str, Any]],
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if len(fold_payloads) != 10:
        raise ValueError("Formal UrbanSound8K cross-validation requires exactly 10 folds.")

    labels = list(range(len(class_names)))
    observed_folds = [int(payload["test_fold"]) for payload in fold_payloads]
    if sorted(observed_folds) != list(range(1, 11)) or len(set(observed_folds)) != 10:
        raise ValueError("Formal test folds must be exactly 1 through 10, each appearing once.")

    fold_rows: list[dict[str, float | int]] = []
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    for payload in sorted(fold_payloads, key=lambda item: int(item["test_fold"])):
        targets = np.asarray(payload["targets"], dtype=np.int64)
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        validate_fold_predictions(targets, probabilities, len(labels))
        predictions = probabilities.argmax(axis=1)
        metrics = classification_metrics(targets.tolist(), predictions.tolist(), labels)
        fold_rows.append(
            {
                "test_fold": int(payload["test_fold"]),
                "validation_fold": int(payload["validation_fold"]),
                "num_examples": int(targets.size),
                **metrics,
            }
        )
        all_targets.append(targets)
        all_probabilities.append(probabilities)

    aggregate_targets = np.concatenate(all_targets)
    aggregate_probabilities = np.concatenate(all_probabilities)
    aggregate_predictions = aggregate_probabilities.argmax(axis=1)
    aggregate_metrics = classification_metrics(
        aggregate_targets.tolist(),
        aggregate_predictions.tolist(),
        labels,
    )

    from sklearn.metrics import f1_score

    per_class_values = f1_score(
        aggregate_targets,
        aggregate_predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )
    summary: dict[str, Any] = {
        "protocol": "fixed cyclic validation fold; each UrbanSound8K fold tested exactly once",
        "folds": fold_rows,
        "aggregate_metrics": aggregate_metrics,
        "per_class_f1": [
            {"class_id": label, "class_name": class_names[label], "f1": float(per_class_values[label])}
            for label in labels
        ],
        "test_evaluated": True,
        "test_folds_evaluated_once": list(range(1, 11)),
    }
    for metric_name in METRIC_NAMES:
        values = np.asarray([float(row[metric_name]) for row in fold_rows], dtype=np.float64)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std(ddof=0))

    matrix = confusion_matrix_array(
        aggregate_targets.tolist(),
        aggregate_predictions.tolist(),
        labels,
    )
    return summary, aggregate_targets, aggregate_probabilities, matrix
