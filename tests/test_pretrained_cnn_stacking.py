from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.stacking import (
    equal_average_probabilities,
    log_probability_features,
    nested_leave_one_fold_out_stacking,
)


ROOT = Path(__file__).resolve().parents[1]


class StackingFeatureTests(unittest.TestCase):
    def test_log_probability_features_concatenate_members(self) -> None:
        first = np.asarray([[0.8, 0.2], [0.1, 0.9]])
        second = np.asarray([[0.6, 0.4], [0.3, 0.7]])
        features = log_probability_features([first, second])
        self.assertEqual(features.shape, (2, 4))
        np.testing.assert_allclose(features[:, :2], np.log(first))
        np.testing.assert_allclose(features[:, 2:], np.log(second))

    def test_probability_helpers_reject_misalignment(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            log_probability_features([np.ones((2, 2)), np.ones((3, 2))])
        with self.assertRaisesRegex(ValueError, "negative"):
            equal_average_probabilities([np.asarray([[1.1, -0.1]])])

    def test_runner_has_no_test_or_fold_selection_cli(self) -> None:
        path = ROOT / "scripts" / "run_pretrained_cnn_stacking_study.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertIn('"test_evaluated": False', source)
        self.assertIn('"formal_test_results_used_for_selection": False', source)
        self.assertNotIn('split="test"', source)
        argument_names = {
            argument.args[0].value
            for argument in ast.walk(tree)
            if isinstance(argument, ast.Call)
            and isinstance(argument.func, ast.Attribute)
            and argument.func.attr == "add_argument"
            and argument.args
            and isinstance(argument.args[0], ast.Constant)
        }
        self.assertNotIn("--folds", argument_names)
        self.assertNotIn("--test-fold", argument_names)


class NestedStackingTests(unittest.TestCase):
    @staticmethod
    def _fold_payload(fold: int) -> dict[str, object]:
        targets = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        probability_sets = []
        for member in range(3):
            confidence = 0.62 + 0.04 * member + 0.01 * fold
            probabilities = np.empty((targets.size, 2), dtype=np.float64)
            probabilities[:, 0] = np.where(targets == 0, confidence, 1.0 - confidence)
            probabilities[:, 1] = 1.0 - probabilities[:, 0]
            probability_sets.append(probabilities)
        return {"targets": targets, "probability_sets": probability_sets}

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_nested_stacking_never_trains_on_outer_fold(self) -> None:
        payloads = {fold: self._fold_payload(fold) for fold in (1, 4, 7)}
        results = nested_leave_one_fold_out_stacking(
            payloads,
            labels=[0, 1],
            c_grid=[0.1, 1.0],
            random_state=42,
        )
        self.assertEqual(set(results), {1, 4, 7})
        for fold, result in results.items():
            self.assertNotIn(fold, result["training_folds"])
            self.assertEqual(len(result["training_folds"]), 2)
            self.assertIn(result["selected_c"], {0.1, 1.0})
            stacked = np.asarray(result["stacked_probabilities"])
            baseline = np.asarray(result["baseline_probabilities"])
            self.assertEqual(stacked.shape, (8, 2))
            np.testing.assert_allclose(stacked.sum(axis=1), 1.0)
            np.testing.assert_allclose(baseline.sum(axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
