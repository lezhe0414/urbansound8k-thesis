from __future__ import annotations

import unittest


try:
    import torch

    from src.data.augmentation import (
        SpectrogramAugmenter,
        SpectrogramBatchMixer,
        _shift_with_zeros,
        augmentation_config_from_training,
        batch_mix_config_from_training,
    )
except Exception as exc:  # pragma: no cover - dependency availability controls skip
    torch = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(torch is None, f"PyTorch unavailable: {IMPORT_ERROR}")
class SpectrogramAugmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.inputs = torch.linspace(-1.0, 1.0, steps=4 * 2 * 16 * 24).reshape(4, 2, 16, 24)
        self.targets = torch.tensor([0, 1, 2, 3])

    def test_disabled_augmenter_preserves_inputs(self) -> None:
        augmenter = SpectrogramAugmenter({"enabled": False})
        output = augmenter(self.inputs)
        self.assertTrue(torch.equal(output, self.inputs))

    def test_shift_uses_zero_padding_instead_of_wrapping(self) -> None:
        sample = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        shifted = _shift_with_zeros(sample, amount=2, dimension=-1)
        expected = torch.tensor([[[0.0, 0.0, 1.0, 2.0]]])
        self.assertTrue(torch.equal(shifted, expected))

    def test_combined_augmentation_preserves_shape_and_finite_values(self) -> None:
        augmenter = SpectrogramAugmenter(
            {
                "time_shift": {"probability": 1.0, "max_frames": 3},
                "frequency_shift": {"probability": 1.0, "max_bins": 2},
                "time_stretch": {"probability": 1.0, "min_rate": 0.9, "max_rate": 1.1},
                "gain": {"probability": 1.0, "min_scale": 0.8, "max_scale": 1.2},
                "noise": {"probability": 1.0, "std": 0.02},
                "spec_augment": {
                    "probability": 1.0,
                    "frequency_mask_param": 3,
                    "time_mask_param": 4,
                    "num_frequency_masks": 1,
                    "num_time_masks": 1,
                },
            }
        )
        output = augmenter(self.inputs)
        self.assertEqual(tuple(output.shape), tuple(self.inputs.shape))
        self.assertTrue(torch.isfinite(output).all())
        self.assertFalse(torch.equal(output, self.inputs))

    def test_mixup_uses_per_sample_lambdas(self) -> None:
        mixer = SpectrogramBatchMixer(
            {"enabled": True, "mode": "mixup", "probability": 1.0, "mixup_alpha": 0.4}
        )
        mixed = mixer(self.inputs, self.targets)
        self.assertEqual(mixed.method, "mixup")
        self.assertEqual(tuple(mixed.inputs.shape), tuple(self.inputs.shape))
        self.assertEqual(tuple(mixed.lambdas.shape), (4,))
        self.assertTrue(torch.all(mixed.lambdas >= 0.5))
        self.assertTrue(torch.all(mixed.lambdas <= 1.0))

    def test_cutmix_preserves_shape_and_replaces_content(self) -> None:
        mixer = SpectrogramBatchMixer(
            {"enabled": True, "mode": "cutmix", "probability": 1.0, "cutmix_alpha": 1.0}
        )
        mixed = mixer(self.inputs, self.targets)
        self.assertEqual(mixed.method, "cutmix")
        self.assertEqual(tuple(mixed.inputs.shape), tuple(self.inputs.shape))
        self.assertTrue(torch.all(mixed.lambdas >= 0.5))
        self.assertFalse(torch.equal(mixed.inputs, self.inputs))

    def test_legacy_configs_remain_supported(self) -> None:
        training = {
            "spec_augment": {"probability": 0.2, "frequency_mask_param": 8},
            "mixup": {"enabled": True, "probability": 0.3, "alpha": 0.1},
        }
        augmentation = augmentation_config_from_training(training)
        batch_mix = batch_mix_config_from_training(training)
        self.assertEqual(augmentation["spec_augment"]["frequency_mask_param"], 8)
        self.assertEqual(batch_mix["mode"], "mixup")
        self.assertAlmostEqual(batch_mix["mixup_alpha"], 0.1)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpectrogramAugmenter({"noise": {"probability": 1.5, "std": 0.1}})

    def test_probability_scale_can_disable_stochastic_regularization(self) -> None:
        augmenter = SpectrogramAugmenter({"noise": {"probability": 1.0, "std": 1.0}})
        augmenter.set_probability_scale(0.0)
        self.assertTrue(torch.equal(augmenter(self.inputs), self.inputs))

        mixer = SpectrogramBatchMixer(
            {"enabled": True, "mode": "mixup", "probability": 1.0, "mixup_alpha": 0.4}
        )
        mixer.set_probability_scale(0.0)
        self.assertEqual(mixer(self.inputs, self.targets).method, "none")

    def test_invalid_probability_scale_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpectrogramAugmenter().set_probability_scale(-0.1)


if __name__ == "__main__":
    unittest.main()
