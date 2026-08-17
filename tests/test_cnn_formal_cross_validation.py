from __future__ import annotations

import ast
import hashlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np

from src.formal_cross_validation import summarize_fold_predictions, validate_fold_predictions


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_cnn_formal_cross_validation.py"
LOCKED_CONFIG = ROOT / "configs" / "cnn_aug_final.yaml"
EXPECTED_SHA256 = "6831eedade7a0cb6e7d2e2b98d32bd067bcc1c7fe62568a2059ead4fe68b82e4"


class FormalProtocolStaticTests(unittest.TestCase):
    def test_locked_config_digest_is_immutable(self) -> None:
        self.assertEqual(hashlib.sha256(LOCKED_CONFIG.read_bytes()).hexdigest(), EXPECTED_SHA256)

    def test_runner_exposes_no_fold_seed_or_hyperparameter_cli(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        arguments = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        }
        self.assertEqual(arguments, {"--output-name", "--backup-root", "--resume"})
        self.assertIn("test_selection_prohibited", source)
        self.assertIn("model_selection_used_test_metrics", source)

    def test_evaluator_persists_probabilities(self) -> None:
        source = (ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
        self.assertIn("evaluation_predictions.npz", source)
        self.assertIn("torch.softmax(logits, dim=1)", source)

    def test_protocol_uses_json_stable_fold_mapping(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(source.count('"validation_fold_mapping": _json_fold_mapping()'), 2)
        self.assertIn("return {str(fold): val_fold", source)


class FormalAggregationTests(unittest.TestCase):
    def test_prediction_validation_rejects_wrong_class_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_fold_predictions(np.asarray([0, 1]), np.ones((2, 3)), num_classes=2)

    def test_prediction_validation_rejects_non_normalized_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_fold_predictions(
                np.asarray([0, 1]),
                np.asarray([[0.8, 0.8], [0.2, 0.2]]),
                num_classes=2,
            )

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_summary_requires_and_aggregates_all_ten_folds(self) -> None:
        payloads = []
        for fold in range(1, 11):
            targets = np.asarray([0, 1], dtype=np.int64)
            probabilities = np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float64)
            payloads.append(
                {
                    "test_fold": fold,
                    "validation_fold": (fold % 10) + 1,
                    "targets": targets,
                    "probabilities": probabilities,
                }
            )

        summary, targets, probabilities, matrix = summarize_fold_predictions(payloads, ["a", "b"])

        self.assertEqual(summary["test_folds_evaluated_once"], list(range(1, 11)))
        self.assertAlmostEqual(summary["f1_macro_mean"], 1.0)
        self.assertAlmostEqual(summary["accuracy_std"], 0.0)
        self.assertEqual(targets.shape, (20,))
        self.assertEqual(probabilities.shape, (20, 2))
        np.testing.assert_array_equal(matrix, np.asarray([[10, 0], [0, 10]]))


if __name__ == "__main__":
    unittest.main()
