from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.data import deterministic_time_shift
from src.utils.metrics import classification_metrics


def tta_offsets_samples(config: dict | None, sample_rate: int) -> list[int]:
    config = dict(config or {})
    if not bool(config.get("enabled", False)):
        return [0]
    offsets_seconds = [float(value) for value in config.get("offsets_seconds", [-0.5, 0.0, 0.5])]
    if not offsets_seconds:
        raise ValueError("evaluation.tta.offsets_seconds must not be empty.")
    offsets = [int(round(value * int(sample_rate))) for value in offsets_seconds]
    if 0 not in offsets:
        raise ValueError("evaluation.tta.offsets_seconds must include 0.0 as the unshifted view.")
    return list(dict.fromkeys(offsets))


def mean_tta_probabilities(
    model: nn.Module,
    waveforms: torch.Tensor,
    offsets_samples: Iterable[int],
    mixed_precision: bool,
) -> torch.Tensor:
    views: list[torch.Tensor] = []
    for offset in offsets_samples:
        shifted = deterministic_time_shift(waveforms, int(offset)) if int(offset) else waveforms
        with torch.autocast(
            device_type=waveforms.device.type,
            dtype=torch.float16,
            enabled=mixed_precision and waveforms.device.type == "cuda",
        ):
            views.append(model(shifted).softmax(dim=1))
    return torch.stack(views, dim=0).mean(dim=0)


def predict_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    labels: list[int],
    criterion: nn.Module | None = None,
    mixed_precision: bool = False,
    offsets_samples: Iterable[int] = (0,),
    description: str = "evaluation",
) -> tuple[float | None, dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true: list[int] = []
    probability_batches: list[np.ndarray] = []
    with torch.no_grad():
        for waveforms, targets in tqdm(loader, desc=description, leave=False):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            probabilities = mean_tta_probabilities(
                model,
                waveforms,
                offsets_samples=offsets_samples,
                mixed_precision=mixed_precision,
            )
            if criterion is not None:
                total_loss += float(criterion(probabilities.clamp_min(1e-8).log(), targets).item()) * waveforms.size(0)
            y_true.extend(targets.cpu().tolist())
            probability_batches.append(probabilities.cpu().numpy())

    targets_array = np.asarray(y_true, dtype=np.int64)
    probabilities_array = np.concatenate(probability_batches, axis=0)
    predictions = probabilities_array.argmax(axis=1)
    metrics = classification_metrics(targets_array.tolist(), predictions.tolist(), labels)
    loss = None if criterion is None else total_loss / max(len(loader.dataset), 1)
    return loss, metrics, targets_array, probabilities_array
