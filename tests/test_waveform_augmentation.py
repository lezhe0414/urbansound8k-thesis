from __future__ import annotations

import unittest


try:
    import torch

    from src.data.waveform_augmentation import (
        WaveformBatchAugmenter,
        deterministic_time_shift,
        zero_fill_shift,
    )
    from src.pretrained_cnn_inference import tta_offsets_samples
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"Audio ML dependencies unavailable: {DEPENDENCY_ERROR}")
class WaveformAugmentationTests(unittest.TestCase):
    def test_zero_fill_shift_never_wraps_audio(self) -> None:
        waveforms = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        shifted = zero_fill_shift(waveforms, torch.tensor([2, -1]))
        self.assertTrue(torch.equal(shifted[0], torch.tensor([0.0, 0.0, 1.0, 2.0])))
        self.assertTrue(torch.equal(shifted[1], torch.tensor([2.0, 3.0, 4.0, 0.0])))

    def test_deterministic_shift_applies_same_view_to_batch(self) -> None:
        waveforms = torch.arange(10, dtype=torch.float32).reshape(2, 5)
        shifted = deterministic_time_shift(waveforms, 1)
        self.assertTrue(torch.equal(shifted[:, 0], torch.zeros(2)))
        self.assertTrue(torch.equal(shifted[:, 1:], waveforms[:, :-1]))

    def test_augmenter_tracks_each_enabled_transform(self) -> None:
        torch.manual_seed(7)
        augmenter = WaveformBatchAugmenter(
            {
                "enabled": True,
                "clamp": True,
                "time_shift": {"enabled": True, "probability": 1.0, "max_seconds": 0.1},
                "gain": {"enabled": True, "probability": 1.0, "min_db": -1.0, "max_db": 1.0},
                "gaussian_noise": {
                    "enabled": True,
                    "probability": 1.0,
                    "min_snr_db": 25.0,
                    "max_snr_db": 30.0,
                },
            },
            sample_rate=100,
        )
        output, stats = augmenter(torch.full((3, 100), 0.25))
        self.assertEqual(tuple(output.shape), (3, 100))
        self.assertEqual((stats.shifted, stats.gained, stats.noised), (3, 3, 3))
        self.assertLessEqual(float(output.abs().max()), 1.0)

    def test_tta_offsets_require_unshifted_view(self) -> None:
        self.assertEqual(tta_offsets_samples({"enabled": False}, 100), [0])
        self.assertEqual(
            tta_offsets_samples({"enabled": True, "offsets_seconds": [-0.5, 0.0, 0.5]}, 100),
            [-50, 0, 50],
        )
        with self.assertRaises(ValueError):
            tta_offsets_samples({"enabled": True, "offsets_seconds": [-0.5, 0.5]}, 100)


if __name__ == "__main__":
    unittest.main()

