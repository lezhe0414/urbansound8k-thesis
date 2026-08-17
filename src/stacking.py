from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


DEFAULT_C_GRID = (0.01, 0.1, 1.0, 10.0)


def log_probability_features(
    probability_sets: Sequence[np.ndarray],
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Concatenate normalized log probabilities from fixed ensemble members."""
    if not probability_sets:
        raise ValueError("At least one member probability array is required.")
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be between zero and one.")

    normalized: list[np.ndarray] = []
    reference_shape: tuple[int, int] | None = None
    for index, values in enumerate(probability_sets):
        probabilities = np.asarray(values, dtype=np.float64)
        if probabilities.ndim != 2:
            raise ValueError(f"Member {index} probabilities must be a 2-D array.")
        if reference_shape is None:
            reference_shape = probabilities.shape
        elif probabilities.shape != reference_shape:
            raise ValueError("All member probability arrays must have identical shapes.")
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Member {index} probabilities contain non-finite values.")
        if (probabilities < 0.0).any():
            raise ValueError(f"Member {index} probabilities contain negative values.")
        row_sums = probabilities.sum(axis=1, keepdims=True)
        if (row_sums <= 0.0).any():
            raise ValueError(f"Member {index} probabilities contain an empty row.")
        normalized.append(probabilities / row_sums)

    return np.concatenate(
        [np.log(np.clip(probabilities, epsilon, 1.0)) for probabilities in normalized],
        axis=1,
    )


def equal_average_probabilities(probability_sets: Sequence[np.ndarray]) -> np.ndarray:
    if not probability_sets:
        raise ValueError("At least one member probability array is required.")
    arrays = [np.asarray(values, dtype=np.float64) for values in probability_sets]
    reference_shape = arrays[0].shape
    if any(values.ndim != 2 or values.shape != reference_shape for values in arrays):
        raise ValueError("All member probability arrays must be aligned 2-D arrays.")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Member probabilities contain non-finite values.")
    if any((values < 0.0).any() for values in arrays):
        raise ValueError("Member probabilities contain negative values.")
    mean_probabilities = np.stack(arrays, axis=0).mean(axis=0)
    row_sums = mean_probabilities.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any() or not np.isfinite(mean_probabilities).all():
        raise ValueError("Averaged probabilities are invalid.")
    return mean_probabilities / row_sums


def _make_estimator(c_value: float, random_state: int):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - dependency failure only
        raise RuntimeError("scikit-learn is required for logistic stacking.") from exc

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=int(random_state),
                ),
            ),
        ]
    )


def _aligned_predict_proba(estimator, features: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    probabilities = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.named_steps["classifier"].classes_, dtype=np.int64)
    aligned = np.zeros((probabilities.shape[0], len(labels)), dtype=np.float64)
    label_positions = {int(label): index for index, label in enumerate(labels)}
    for column, class_label in enumerate(classes):
        if int(class_label) not in label_positions:
            raise ValueError(f"Unexpected stacker class {class_label}.")
        aligned[:, label_positions[int(class_label)]] = probabilities[:, column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if (row_sums <= 0.0).any():
        raise ValueError("Stacker returned an empty probability row.")
    return aligned / row_sums


def _macro_f1(targets: np.ndarray, probabilities: np.ndarray, labels: Sequence[int]) -> float:
    try:
        from sklearn.metrics import f1_score
    except ImportError as exc:  # pragma: no cover - dependency failure only
        raise RuntimeError("scikit-learn is required for logistic stacking.") from exc
    predictions = probabilities.argmax(axis=1)
    return float(
        f1_score(
            targets,
            predictions,
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


def _concatenate_folds(
    fold_features: Mapping[int, np.ndarray],
    fold_targets: Mapping[int, np.ndarray],
    folds: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = [int(fold) for fold in folds]
    if not ordered:
        raise ValueError("At least one training fold is required.")
    return (
        np.concatenate([fold_features[fold] for fold in ordered], axis=0),
        np.concatenate([fold_targets[fold] for fold in ordered], axis=0),
    )


def nested_leave_one_fold_out_stacking(
    fold_payloads: Mapping[int, Mapping[str, object]],
    labels: Sequence[int],
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    random_state: int = 42,
) -> dict[int, dict[str, object]]:
    """Fit nested logistic stackers without using the outer target labels for selection."""
    folds = sorted(int(fold) for fold in fold_payloads)
    if len(folds) < 3:
        raise ValueError("Nested leave-one-fold-out stacking requires at least three folds.")
    c_values = tuple(float(value) for value in c_grid)
    if not c_values or any(value <= 0.0 for value in c_values):
        raise ValueError("All C values must be positive.")
    if len(set(c_values)) != len(c_values):
        raise ValueError("C values must be unique.")
    label_values = tuple(int(label) for label in labels)
    if not label_values:
        raise ValueError("At least one class label is required.")

    fold_targets: dict[int, np.ndarray] = {}
    fold_probability_sets: dict[int, list[np.ndarray]] = {}
    fold_features: dict[int, np.ndarray] = {}
    expected_member_count: int | None = None
    expected_class_count = len(label_values)
    for fold in folds:
        payload = fold_payloads[fold]
        targets = np.asarray(payload["targets"], dtype=np.int64)
        probability_sets = [
            np.asarray(values, dtype=np.float64)
            for values in payload["probability_sets"]  # type: ignore[index]
        ]
        if targets.ndim != 1:
            raise ValueError(f"Fold {fold} targets must be a 1-D array.")
        if expected_member_count is None:
            expected_member_count = len(probability_sets)
        elif len(probability_sets) != expected_member_count:
            raise ValueError("Every fold must contain the same number of ensemble members.")
        if not probability_sets:
            raise ValueError(f"Fold {fold} has no member probabilities.")
        if any(values.shape != (targets.size, expected_class_count) for values in probability_sets):
            raise ValueError(f"Fold {fold} probabilities do not align with targets/classes.")
        if not set(targets.tolist()).issubset(set(label_values)):
            raise ValueError(f"Fold {fold} contains an unknown target label.")
        fold_targets[fold] = targets
        fold_probability_sets[fold] = probability_sets
        fold_features[fold] = log_probability_features(probability_sets)

    results: dict[int, dict[str, object]] = {}
    for outer_fold in folds:
        training_folds = [fold for fold in folds if fold != outer_fold]
        inner_scores: dict[float, list[float]] = {}
        for c_value in c_values:
            scores: list[float] = []
            for inner_validation_fold in training_folds:
                inner_training_folds = [
                    fold for fold in training_folds if fold != inner_validation_fold
                ]
                train_features, train_targets = _concatenate_folds(
                    fold_features,
                    fold_targets,
                    inner_training_folds,
                )
                estimator = _make_estimator(c_value, random_state)
                estimator.fit(train_features, train_targets)
                probabilities = _aligned_predict_proba(
                    estimator,
                    fold_features[inner_validation_fold],
                    label_values,
                )
                scores.append(
                    _macro_f1(
                        fold_targets[inner_validation_fold],
                        probabilities,
                        label_values,
                    )
                )
            inner_scores[c_value] = scores

        selected_c = min(
            c_values,
            key=lambda value: (-float(np.mean(inner_scores[value])), value),
        )
        train_features, train_targets = _concatenate_folds(
            fold_features,
            fold_targets,
            training_folds,
        )
        estimator = _make_estimator(selected_c, random_state)
        estimator.fit(train_features, train_targets)
        stacked_probabilities = _aligned_predict_proba(
            estimator,
            fold_features[outer_fold],
            label_values,
        )
        baseline_probabilities = equal_average_probabilities(
            fold_probability_sets[outer_fold]
        )
        scaler = estimator.named_steps["scaler"]
        classifier = estimator.named_steps["classifier"]
        results[outer_fold] = {
            "targets": fold_targets[outer_fold],
            "stacked_probabilities": stacked_probabilities,
            "baseline_probabilities": baseline_probabilities,
            "training_folds": training_folds,
            "selected_c": float(selected_c),
            "inner_scores": {
                str(c_value): {
                    "fold_scores": inner_scores[c_value],
                    "mean_f1_macro": float(np.mean(inner_scores[c_value])),
                }
                for c_value in c_values
            },
            "model_parameters": {
                "classes": classifier.classes_.astype(int).tolist(),
                "coefficients": classifier.coef_.tolist(),
                "intercepts": classifier.intercept_.tolist(),
                "iterations": classifier.n_iter_.astype(int).tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
        }
    return results
