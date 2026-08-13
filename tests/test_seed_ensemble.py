from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch

    from src.ensemble import (
        _validate_existing_seed_run,
        average_probabilities,
        normalize_seeds,
        seed_config,
    )
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class SeedEnsembleTests(unittest.TestCase):
    def test_seed_config_disables_ema_and_individual_test(self) -> None:
        base = {
            "run_name": "cnn_final",
            "seed": 1,
            "model": {"name": "cnn"},
            "training": {"ema": {"enabled": True, "decay": 0.995}},
            "evaluation": {"run_test": True},
        }

        config = seed_config(base, 123)

        self.assertEqual(config["run_name"], "cnn_final_seed123")
        self.assertEqual(config["seed"], 123)
        self.assertFalse(config["training"]["ema"]["enabled"])
        self.assertFalse(config["evaluation"]["run_test"])
        self.assertTrue(base["training"]["ema"]["enabled"])

    def test_average_probabilities_uses_arithmetic_mean(self) -> None:
        first = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
        second = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
        third = torch.tensor([[0.7, 0.3], [0.3, 0.7]])

        averaged = average_probabilities([first, second, third])

        expected = torch.tensor([[0.7, 0.3], [0.3, 0.7]])
        self.assertTrue(torch.allclose(averaged, expected))

    def test_exactly_three_unique_seeds_are_required(self) -> None:
        self.assertEqual(normalize_seeds([42, 123, 2026]), (42, 123, 2026))
        with self.assertRaises(ValueError):
            normalize_seeds([42, 123])
        with self.assertRaises(ValueError):
            normalize_seeds([42, 42, 123])

    def test_probability_shapes_must_match(self) -> None:
        with self.assertRaises(ValueError):
            average_probabilities([torch.ones(2, 3), torch.ones(3, 3)])

    def test_existing_seed_run_must_match_locked_config(self) -> None:
        expected = {
            "run_name": "cnn_final_seed42",
            "seed": 42,
            "model": {"name": "cnn"},
            "training": {"ema": {"enabled": False}, "learning_rate": 0.00045},
            "evaluation": {"run_test": False},
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for filename in ("best_model.pt", "validation_metrics.json"):
                (run_dir / filename).touch()
            resolved = {"config": expected, "fold": 10, "val_fold": 1}
            (run_dir / "config_resolved.json").write_text(json.dumps(resolved), encoding="utf-8")

            _validate_existing_seed_run(run_dir, expected, fold=10)

            changed = {**expected, "training": {**expected["training"], "learning_rate": 0.001}}
            with self.assertRaises(ValueError):
                _validate_existing_seed_run(run_dir, changed, fold=10)


if __name__ == "__main__":
    unittest.main()
