from __future__ import annotations

import copy
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.run_cnn_controlled_search import (
    INITIAL_CONFIGS,
    MUTATIONS,
    ROOT,
    _validation_metrics,
)
from src.utils.config import load_config

try:
    import yaml  # noqa: F401
except ImportError:
    YAML_AVAILABLE = False
else:
    YAML_AVAILABLE = True


class ControlledSearchTests(unittest.TestCase):
    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML is not installed")
    def test_initial_profiles_keep_non_augmentation_conditions_fixed(self) -> None:
        configs = [load_config(ROOT / path) for path in INITIAL_CONFIGS]
        reference = copy.deepcopy(configs[0])
        reference.pop("run_name")
        reference["training"].pop("augmentation")
        reference["training"].pop("batch_mix")

        for config in configs:
            candidate = copy.deepcopy(config)
            candidate.pop("run_name")
            candidate["training"].pop("augmentation")
            candidate["training"].pop("batch_mix")
            self.assertEqual(candidate, reference)
            self.assertFalse(config["evaluation"]["run_test"])

    def test_validation_selection_uses_highest_macro_f1(self) -> None:
        rows = [
            {
                "epoch": 1,
                "train_accuracy": 0.70,
                "train_f1_macro": 0.69,
                "train_loss": 0.80,
                "val_accuracy": 0.91,
                "val_f1_macro": 0.60,
                "val_loss": 0.70,
            },
            {
                "epoch": 2,
                "train_accuracy": 0.75,
                "train_f1_macro": 0.74,
                "train_loss": 0.70,
                "val_accuracy": 0.80,
                "val_f1_macro": 0.72,
                "val_loss": 0.75,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            with (run_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            metrics = _validation_metrics(run_dir)

        self.assertEqual(metrics["best_epoch"], 2)
        self.assertAlmostEqual(metrics["val_f1_macro"], 0.72)
        self.assertAlmostEqual(metrics["val_accuracy"], 0.80)

    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML is not installed")
    def test_each_iteration_mutates_one_declared_category(self) -> None:
        base = load_config(ROOT / "configs/cnn_aug_balanced.yaml")
        for mutation in MUTATIONS:
            candidate = copy.deepcopy(base)
            changed_value = mutation.apply(candidate)
            self.assertIsInstance(changed_value, str)
            self.assertNotEqual(candidate, base, mutation.name)


if __name__ == "__main__":
    unittest.main()
