from __future__ import annotations

import math


def regularization_scale(config: dict, epoch: int, epochs: int) -> float:
    """Return the scheduled strength for stochastic training regularization."""
    if not config or not bool(config.get("enabled", False)):
        return 1.0

    name = str(config.get("name", "linear")).lower()
    if name not in {"linear", "cosine"}:
        raise ValueError("Regularization schedule name must be linear or cosine.")
    start_epoch = int(config.get("start_epoch", max(1, epochs // 2)))
    end_epoch = int(config.get("end_epoch", epochs))
    final_scale = float(config.get("final_scale", 0.0))
    if start_epoch < 1 or end_epoch < start_epoch or end_epoch > epochs:
        raise ValueError("Regularization schedule requires 1 <= start_epoch <= end_epoch <= epochs.")
    if not 0.0 <= final_scale <= 1.0:
        raise ValueError("Regularization schedule final_scale must be in [0, 1].")
    if epoch <= start_epoch:
        return 1.0
    if epoch >= end_epoch:
        return final_scale

    progress = (epoch - start_epoch) / float(end_epoch - start_epoch)
    if name == "cosine":
        progress = 0.5 * (1.0 - math.cos(math.pi * progress))
    return 1.0 + (final_scale - 1.0) * progress
