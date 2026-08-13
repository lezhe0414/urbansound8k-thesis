from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_cnn_breakthrough_search import (
    DEFAULT_CONFIGS,
    DEFAULT_VALIDATION_FOLDS,
    ROOT,
    _aggregate,
    _candidate_config,
    _validation_offset,
    _write_progress,
)
from src.utils.config import load_config
from src.utils.schedules import regularization_scale

try:
    import yaml  # noqa: F401
except ImportError:
    YAML_AVAILABLE = False
else:
    YAML_AVAILABLE = True


class BreakthroughSearchTests(unittest.TestCase):
    def test_development_folds_leave_fold_10_locked(self) -> None:
        for validation_fold in DEFAULT_VALIDATION_FOLDS:
            offset = _validation_offset(10, validation_fold)
            self.assertEqual(((10 - 1 + offset) % 10) + 1, validation_fold)

    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML is not installed")
    def test_candidate_config_disables_test_evaluation(self) -> None:
        base = load_config(ROOT / DEFAULT_CONFIGS[0])
        candidate = _candidate_config(base, validation_fold=4, locked_test_fold=10)
        self.assertFalse(candidate["evaluation"]["run_test"])
        self.assertEqual(candidate["data"]["val_fold_offset"], 4)
        self.assertTrue(candidate["run_name"].endswith("_devval4"))

    @unittest.skipUnless(YAML_AVAILABLE, "PyYAML is not installed")
    def test_all_breakthrough_configs_are_validation_only(self) -> None:
        for relative_path in DEFAULT_CONFIGS:
            config = load_config(ROOT / relative_path)
            self.assertFalse(config["evaluation"]["run_test"], relative_path)

    def test_ranking_uses_mean_validation_macro_f1(self) -> None:
        rows = [
            {"candidate": "stable", "validation_fold": 1, "val_f1_macro": 0.82, "val_accuracy": 0.8, "duration_seconds": 1},
            {"candidate": "stable", "validation_fold": 4, "val_f1_macro": 0.84, "val_accuracy": 0.8, "duration_seconds": 1},
            {"candidate": "spiky", "validation_fold": 1, "val_f1_macro": 0.90, "val_accuracy": 0.8, "duration_seconds": 1},
            {"candidate": "spiky", "validation_fold": 4, "val_f1_macro": 0.70, "val_accuracy": 0.8, "duration_seconds": 1},
        ]
        ranking = _aggregate(rows)
        self.assertEqual(ranking[0]["candidate"], "stable")
        self.assertAlmostEqual(ranking[0]["mean_val_f1_macro"], 0.83)

    def test_cosine_regularization_cooldown(self) -> None:
        config = {
            "enabled": True,
            "name": "cosine",
            "start_epoch": 6,
            "end_epoch": 10,
            "final_scale": 0.1,
        }
        self.assertEqual(regularization_scale(config, 6, 10), 1.0)
        self.assertAlmostEqual(regularization_scale(config, 10, 10), 0.1)
        self.assertGreater(regularization_scale(config, 7, 10), regularization_scale(config, 9, 10))

    def test_progress_is_written_after_each_run(self) -> None:
        rows = [{"candidate": "control", "val_f1_macro": 0.8}]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "progress.csv"
            json_path = root / "progress.json"
            _write_progress(rows, csv_path, json_path, backup_dir=None)
            self.assertTrue(csv_path.exists())
            self.assertEqual(__import__("json").loads(json_path.read_text(encoding="utf-8")), rows)


if __name__ == "__main__":
    unittest.main()
