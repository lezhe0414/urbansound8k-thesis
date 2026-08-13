from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintain a smoothed, evaluation-only copy of a model."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in the range [0, 1).")

        self.decay = float(decay)
        self.num_updates = 0
        self.model = deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source_state = model.state_dict()
        ema_state = self.model.state_dict()
        if source_state.keys() != ema_state.keys():
            raise ValueError("EMA model structure does not match the training model.")

        if self.num_updates == 0:
            self.model.load_state_dict(source_state)
        else:
            update_weight = 1.0 - self.decay
            for name, ema_value in ema_state.items():
                source_value = source_state[name].detach()
                if torch.is_floating_point(ema_value):
                    ema_value.lerp_(source_value, update_weight)
                else:
                    ema_value.copy_(source_value)

        self.num_updates += 1
