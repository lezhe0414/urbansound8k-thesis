from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class WaveformAugmentationStats:
    shifted: int = 0
    gained: int = 0
    noised: int = 0


def zero_fill_shift(waveforms: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Shift each waveform without wrapping samples across clip boundaries."""
    if waveforms.ndim != 2:
        raise ValueError(f"Expected waveform shape [batch, samples], received {tuple(waveforms.shape)}")
    if offsets.ndim != 1 or offsets.numel() != waveforms.size(0):
        raise ValueError("offsets must contain one integer shift per waveform.")

    shifted = torch.zeros_like(waveforms)
    sample_count = waveforms.size(1)
    for index, raw_offset in enumerate(offsets.tolist()):
        offset = int(raw_offset)
        if offset == 0:
            shifted[index] = waveforms[index]
        elif 0 < offset < sample_count:
            shifted[index, offset:] = waveforms[index, : sample_count - offset]
        elif -sample_count < offset < 0:
            shifted[index, : sample_count + offset] = waveforms[index, -offset:]
    return shifted


def deterministic_time_shift(waveforms: torch.Tensor, offset_samples: int) -> torch.Tensor:
    offsets = torch.full(
        (waveforms.size(0),),
        int(offset_samples),
        dtype=torch.long,
        device=waveforms.device,
    )
    return zero_fill_shift(waveforms, offsets)


class WaveformBatchAugmenter:
    """Apply independent waveform transforms after cache loading and before the official frontend."""

    def __init__(self, config: dict | None, sample_rate: int) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.sample_rate = int(sample_rate)
        self.shift_config = dict(self.config.get("time_shift") or {})
        self.gain_config = dict(self.config.get("gain") or {})
        self.noise_config = dict(self.config.get("gaussian_noise") or {})
        self.clamp = bool(self.config.get("clamp", True))
        self._validate()

    @staticmethod
    def _probability(config: dict) -> float:
        return float(config.get("probability", 0.0)) if bool(config.get("enabled", False)) else 0.0

    def _validate(self) -> None:
        for name, config in (
            ("time_shift", self.shift_config),
            ("gain", self.gain_config),
            ("gaussian_noise", self.noise_config),
        ):
            probability = self._probability(config)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"training.waveform_augmentation.{name}.probability must be between 0 and 1.")
        if self._probability(self.shift_config) > 0.0 and float(
            self.shift_config.get("max_seconds", 0.0)
        ) <= 0.0:
            raise ValueError("time_shift.max_seconds must be positive when time shift is enabled.")
        if self._probability(self.gain_config) > 0.0:
            minimum = float(self.gain_config.get("min_db", -3.0))
            maximum = float(self.gain_config.get("max_db", 3.0))
            if minimum > maximum:
                raise ValueError("gain.min_db must not exceed gain.max_db.")
        if self._probability(self.noise_config) > 0.0:
            minimum = float(self.noise_config.get("min_snr_db", 20.0))
            maximum = float(self.noise_config.get("max_snr_db", 35.0))
            if minimum <= 0.0 or minimum > maximum:
                raise ValueError("gaussian_noise SNR bounds must be positive and ordered.")

    @staticmethod
    def _selection(batch_size: int, probability: float, device: torch.device) -> torch.Tensor:
        return torch.rand(batch_size, device=device) < probability

    def __call__(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, WaveformAugmentationStats]:
        if not self.enabled:
            return waveforms, WaveformAugmentationStats()
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, samples], received {tuple(waveforms.shape)}")

        augmented = waveforms.clone()
        batch_size = augmented.size(0)
        shifted_count = 0
        gained_count = 0
        noised_count = 0

        shift_probability = self._probability(self.shift_config)
        if shift_probability > 0.0:
            selected = self._selection(batch_size, shift_probability, augmented.device)
            max_samples = int(round(float(self.shift_config["max_seconds"]) * self.sample_rate))
            offsets = torch.zeros(batch_size, dtype=torch.long, device=augmented.device)
            selected_count = int(selected.sum().item())
            if selected_count:
                offsets[selected] = torch.randint(
                    -max_samples,
                    max_samples + 1,
                    (selected_count,),
                    device=augmented.device,
                )
                augmented = zero_fill_shift(augmented, offsets)
                shifted_count = selected_count

        gain_probability = self._probability(self.gain_config)
        if gain_probability > 0.0:
            selected = self._selection(batch_size, gain_probability, augmented.device)
            selected_count = int(selected.sum().item())
            if selected_count:
                minimum = float(self.gain_config.get("min_db", -3.0))
                maximum = float(self.gain_config.get("max_db", 3.0))
                gains_db = torch.empty(selected_count, device=augmented.device).uniform_(minimum, maximum)
                augmented[selected] *= torch.pow(10.0, gains_db / 20.0).unsqueeze(1)
                gained_count = selected_count

        noise_probability = self._probability(self.noise_config)
        if noise_probability > 0.0:
            selected = self._selection(batch_size, noise_probability, augmented.device)
            selected_count = int(selected.sum().item())
            if selected_count:
                minimum = float(self.noise_config.get("min_snr_db", 20.0))
                maximum = float(self.noise_config.get("max_snr_db", 35.0))
                snr_db = torch.empty(selected_count, device=augmented.device).uniform_(minimum, maximum)
                selected_waveforms = augmented[selected]
                signal_rms = selected_waveforms.square().mean(dim=1).sqrt().clamp_min(1e-5)
                noise_rms = signal_rms / torch.pow(10.0, snr_db / 20.0)
                noise = torch.randn_like(selected_waveforms)
                noise = noise / noise.square().mean(dim=1).sqrt().clamp_min(1e-5).unsqueeze(1)
                augmented[selected] = selected_waveforms + noise * noise_rms.unsqueeze(1)
                noised_count = selected_count

        if self.clamp:
            augmented = augmented.clamp(-1.0, 1.0)
        return augmented, WaveformAugmentationStats(
            shifted=shifted_count,
            gained=gained_count,
            noised=noised_count,
        )
