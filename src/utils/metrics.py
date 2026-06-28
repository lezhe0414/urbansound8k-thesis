from __future__ import annotations

from pathlib import Path

import numpy as np


def classification_metrics(y_true: list[int], y_pred: list[int], labels: list[int]) -> dict[str, float]:
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def confusion_matrix_array(y_true: list[int], y_pred: list[int], labels: list[int]) -> np.ndarray:
    try:
        from sklearn.metrics import confusion_matrix
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise RuntimeError("scikit-learn is required. Install dependencies with `pip install -r requirements.txt`.") from exc
    return confusion_matrix(y_true, y_pred, labels=labels)


def write_history_csv(rows: list[dict], path: str | Path) -> None:
    import csv

    path = Path(path)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
