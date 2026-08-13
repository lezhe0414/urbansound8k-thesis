from __future__ import annotations

import unittest


try:
    import torch
    from torch import nn

    from src.utils.ema import ExponentialMovingAverage
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class ExponentialMovingAverageTests(unittest.TestCase):
    def test_first_update_copies_trained_weights(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        ema = ExponentialMovingAverage(model, decay=0.9)
        with torch.no_grad():
            model.weight.fill_(2.0)

        ema.update(model)

        self.assertTrue(torch.equal(ema.model.weight, model.weight))
        self.assertEqual(ema.num_updates, 1)
        self.assertFalse(ema.model.weight.requires_grad)

    def test_later_updates_apply_exponential_smoothing(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        ema = ExponentialMovingAverage(model, decay=0.75)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)
        with torch.no_grad():
            model.weight.fill_(6.0)

        ema.update(model)

        self.assertTrue(torch.allclose(ema.model.weight, torch.full_like(model.weight, 3.0)))
        self.assertEqual(ema.num_updates, 2)

    def test_non_floating_buffers_are_copied(self) -> None:
        model = nn.BatchNorm1d(2)
        ema = ExponentialMovingAverage(model, decay=0.9)
        with torch.no_grad():
            model.num_batches_tracked.fill_(7)
        ema.update(model)
        with torch.no_grad():
            model.num_batches_tracked.fill_(11)

        ema.update(model)

        self.assertEqual(int(ema.model.num_batches_tracked.item()), 11)

    def test_invalid_decay_is_rejected(self) -> None:
        model = nn.Linear(2, 1)
        with self.assertRaises(ValueError):
            ExponentialMovingAverage(model, decay=1.0)


if __name__ == "__main__":
    unittest.main()
