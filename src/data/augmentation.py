from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _probability(config: dict, default: float = 0.0) -> float:
    value = float(config.get("probability", default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Augmentation probability must be in [0, 1], got {value}.")
    return value


def _positive_int(config: dict, key: str, default: int = 0) -> int:
    value = int(config.get(key, default))
    if value < 0:
        raise ValueError(f"{key} must be non-negative, got {value}.")
    return value


def _shift_with_zeros(sample: torch.Tensor, amount: int, dimension: int) -> torch.Tensor:
    """Shift one spectrogram without wrapping content around the opposite edge."""
    if amount == 0:
        return sample

    size = sample.shape[dimension]
    if abs(amount) >= size:
        return torch.zeros_like(sample)

    shifted = torch.roll(sample, shifts=amount, dims=dimension)
    slices = [slice(None)] * sample.ndim
    if amount > 0:
        slices[dimension] = slice(0, amount)
    else:
        slices[dimension] = slice(size + amount, size)
    shifted[tuple(slices)] = 0.0
    return shifted


class SpectrogramAugmenter:
    """Apply configurable, training-only augmentation to normalized spectrograms."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.probability_scale = 1.0
        self._validate()

    def _validate(self) -> None:
        for name in ("time_shift", "frequency_shift", "time_stretch", "gain", "noise", "spec_augment"):
            _probability(self.config.get(name, {}))

        stretch_config = self.config.get("time_stretch", {})
        minimum = float(stretch_config.get("min_rate", 1.0))
        maximum = float(stretch_config.get("max_rate", 1.0))
        if minimum <= 0.0 or maximum <= 0.0 or minimum > maximum:
            raise ValueError("time_stretch rates must be positive and min_rate <= max_rate.")

        gain_config = self.config.get("gain", {})
        minimum = float(gain_config.get("min_scale", 1.0))
        maximum = float(gain_config.get("max_scale", 1.0))
        if minimum <= 0.0 or maximum <= 0.0 or minimum > maximum:
            raise ValueError("gain scales must be positive and min_scale <= max_scale.")

        noise_std = float(self.config.get("noise", {}).get("std", 0.0))
        if noise_std < 0.0:
            raise ValueError("noise std must be non-negative.")

    def set_probability_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Probability scale must be in [0, 1], got {value}.")
        self.probability_scale = float(value)

    def _applies(self, config: dict, device: torch.device) -> bool:
        probability = _probability(config) * self.probability_scale
        return probability > 0.0 and bool(torch.rand((), device=device) < probability)

    @staticmethod
    def _uniform(low: float, high: float, device: torch.device) -> float:
        if low == high:
            return low
        return float(torch.empty((), device=device).uniform_(low, high).item())

    @staticmethod
    def _random_shift(maximum: int, device: torch.device) -> int:
        if maximum <= 0:
            return 0
        return int(torch.randint(-maximum, maximum + 1, (), device=device).item())

    @staticmethod
    def _restore_time_length(sample: torch.Tensor, target_steps: int) -> torch.Tensor:
        current_steps = sample.shape[-1]
        if current_steps == target_steps:
            return sample
        if current_steps > target_steps:
            start = (current_steps - target_steps) // 2
            return sample[..., start : start + target_steps]

        missing = target_steps - current_steps
        left = missing // 2
        right = missing - left
        return F.pad(sample, (left, right), value=0.0)

    def _time_stretch(self, sample: torch.Tensor, config: dict) -> torch.Tensor:
        rate = self._uniform(
            float(config.get("min_rate", 1.0)),
            float(config.get("max_rate", 1.0)),
            sample.device,
        )
        target_steps = sample.shape[-1]
        stretched_steps = max(1, int(round(target_steps / rate)))
        stretched = F.interpolate(
            sample.unsqueeze(0),
            size=(sample.shape[-2], stretched_steps),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return self._restore_time_length(stretched, target_steps)

    @staticmethod
    def _mask_axis(sample: torch.Tensor, axis: int, maximum_width: int, count: int) -> torch.Tensor:
        axis_size = sample.shape[axis]
        maximum_width = min(maximum_width, max(axis_size - 1, 0))
        if maximum_width <= 0 or count <= 0:
            return sample

        for _ in range(count):
            width = int(torch.randint(1, maximum_width + 1, (), device=sample.device).item())
            start = int(torch.randint(0, axis_size - width + 1, (), device=sample.device).item())
            slices = [slice(None)] * sample.ndim
            slices[axis] = slice(start, start + width)
            sample[tuple(slices)] = 0.0
        return sample

    def _spec_augment(self, sample: torch.Tensor, config: dict) -> torch.Tensor:
        sample = sample.clone()
        sample = self._mask_axis(
            sample,
            axis=-2,
            maximum_width=_positive_int(config, "frequency_mask_param"),
            count=_positive_int(config, "num_frequency_masks", 1),
        )
        return self._mask_axis(
            sample,
            axis=-1,
            maximum_width=_positive_int(config, "time_mask_param"),
            count=_positive_int(config, "num_time_masks", 1),
        )

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return inputs
        if inputs.ndim != 4:
            raise ValueError(f"Expected [batch, channels, frequency, time], got {tuple(inputs.shape)}.")

        augmented = inputs.clone()
        for index in range(augmented.shape[0]):
            sample = augmented[index]

            stretch_config = self.config.get("time_stretch", {})
            if self._applies(stretch_config, sample.device):
                sample = self._time_stretch(sample, stretch_config)

            time_shift_config = self.config.get("time_shift", {})
            if self._applies(time_shift_config, sample.device):
                amount = self._random_shift(
                    _positive_int(time_shift_config, "max_frames"),
                    sample.device,
                )
                sample = _shift_with_zeros(sample, amount, dimension=-1)

            frequency_shift_config = self.config.get("frequency_shift", {})
            if self._applies(frequency_shift_config, sample.device):
                amount = self._random_shift(
                    _positive_int(frequency_shift_config, "max_bins"),
                    sample.device,
                )
                sample = _shift_with_zeros(sample, amount, dimension=-2)

            gain_config = self.config.get("gain", {})
            if self._applies(gain_config, sample.device):
                scale = self._uniform(
                    float(gain_config.get("min_scale", 1.0)),
                    float(gain_config.get("max_scale", 1.0)),
                    sample.device,
                )
                sample = sample * scale

            noise_config = self.config.get("noise", {})
            if self._applies(noise_config, sample.device):
                sample = sample + torch.randn_like(sample) * float(noise_config.get("std", 0.0))

            spec_augment_config = self.config.get("spec_augment", {})
            if self._applies(spec_augment_config, sample.device):
                sample = self._spec_augment(sample, spec_augment_config)

            augmented[index] = sample
        return augmented


@dataclass(frozen=True)
class MixedBatch:
    inputs: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lambdas: torch.Tensor
    method: str

    @property
    def metric_targets(self) -> torch.Tensor:
        return self.targets_a


class SpectrogramBatchMixer:
    """Apply per-sample Mixup or spectrogram CutMix after regular augmentation."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.probability = _probability(self.config, default=1.0)
        self.probability_scale = 1.0
        self.mode = str(self.config.get("mode", "mixup")).lower()
        if self.mode not in {"mixup", "cutmix", "random"}:
            raise ValueError("batch_mix mode must be mixup, cutmix, or random.")
        self.mixup_alpha = float(self.config.get("mixup_alpha", self.config.get("alpha", 0.2)))
        self.cutmix_alpha = float(self.config.get("cutmix_alpha", 1.0))
        self.mixup_probability = _probability(
            {"probability": self.config.get("mixup_probability", 0.5)},
            default=0.5,
        )
        if self.mixup_alpha <= 0.0 or self.cutmix_alpha <= 0.0:
            raise ValueError("Mixup and CutMix alpha values must be positive.")

    def set_probability_scale(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Probability scale must be in [0, 1], got {value}.")
        self.probability_scale = float(value)

    @staticmethod
    def _beta(alpha: float, size: int, device: torch.device) -> torch.Tensor:
        concentration = torch.tensor(alpha, device=device)
        values = torch.distributions.Beta(concentration, concentration).sample((size,))
        return torch.maximum(values, 1.0 - values)

    def _mixup(self, inputs: torch.Tensor, permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lambdas = self._beta(self.mixup_alpha, inputs.shape[0], inputs.device)
        view_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mixed = lambdas.view(view_shape) * inputs + (1.0 - lambdas).view(view_shape) * inputs[permutation]
        return mixed, lambdas

    def _cutmix(self, inputs: torch.Tensor, permutation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed = inputs.clone()
        requested_lambdas = self._beta(self.cutmix_alpha, inputs.shape[0], inputs.device)
        actual_lambdas = torch.ones_like(requested_lambdas)
        frequency_bins, time_steps = inputs.shape[-2:]

        for index, requested_lambda in enumerate(requested_lambdas):
            cut_ratio = torch.sqrt(1.0 - requested_lambda)
            cut_frequency = max(1, int(frequency_bins * float(cut_ratio)))
            cut_time = max(1, int(time_steps * float(cut_ratio)))
            center_frequency = int(torch.randint(0, frequency_bins, (), device=inputs.device).item())
            center_time = int(torch.randint(0, time_steps, (), device=inputs.device).item())
            frequency_start = max(0, center_frequency - cut_frequency // 2)
            frequency_end = min(frequency_bins, frequency_start + cut_frequency)
            time_start = max(0, center_time - cut_time // 2)
            time_end = min(time_steps, time_start + cut_time)
            mixed[index, :, frequency_start:frequency_end, time_start:time_end] = inputs[
                permutation[index], :, frequency_start:frequency_end, time_start:time_end
            ]
            area = (frequency_end - frequency_start) * (time_end - time_start)
            actual_lambdas[index] = 1.0 - area / float(frequency_bins * time_steps)

        return mixed, actual_lambdas

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> MixedBatch:
        no_mix = MixedBatch(
            inputs=inputs,
            targets_a=targets,
            targets_b=targets,
            lambdas=torch.ones(inputs.shape[0], device=inputs.device),
            method="none",
        )
        effective_probability = self.probability * self.probability_scale
        if not self.enabled or inputs.shape[0] < 2 or effective_probability <= 0.0:
            return no_mix
        if bool(torch.rand((), device=inputs.device) > effective_probability):
            return no_mix

        method = self.mode
        if method == "random":
            method = "mixup" if bool(torch.rand((), device=inputs.device) < self.mixup_probability) else "cutmix"
        permutation = torch.randperm(inputs.shape[0], device=inputs.device)
        if method == "mixup":
            mixed_inputs, lambdas = self._mixup(inputs, permutation)
        else:
            mixed_inputs, lambdas = self._cutmix(inputs, permutation)
        return MixedBatch(mixed_inputs, targets, targets[permutation], lambdas, method)


def augmentation_config_from_training(training_config: dict) -> dict:
    """Read the new augmentation block while preserving legacy SpecAugment configs."""
    if "augmentation" in training_config:
        return dict(training_config.get("augmentation") or {})
    legacy = dict(training_config.get("spec_augment") or {})
    if not legacy:
        return {"enabled": False}
    return {"enabled": True, "spec_augment": legacy}


def batch_mix_config_from_training(training_config: dict) -> dict:
    """Read the new batch_mix block while preserving legacy Mixup configs."""
    if "batch_mix" in training_config:
        return dict(training_config.get("batch_mix") or {})
    legacy = dict(training_config.get("mixup") or {})
    if not legacy:
        return {"enabled": False}
    return {
        "enabled": bool(legacy.get("enabled", False)),
        "mode": "mixup",
        "probability": float(legacy.get("probability", 1.0)),
        "mixup_alpha": float(legacy.get("alpha", 0.2)),
    }
