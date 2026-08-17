from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalCrossEntropyLoss(nn.Module):
    """Class-weighted multiclass focal loss with CE-compatible normalization."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError("Focal loss gamma must be non-negative.")
        self.gamma = float(gamma)
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", weight.detach().clone())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
        )
        probabilities = F.softmax(logits, dim=1)
        target_probability = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        losses = (1.0 - target_probability).pow(self.gamma) * cross_entropy
        if self.weight is None:
            return losses.mean()
        denominator = self.weight[targets].sum().clamp_min(torch.finfo(losses.dtype).eps)
        return losses.sum() / denominator


def build_classification_loss(
    training_config: dict,
    class_weights: torch.Tensor | None,
) -> nn.Module:
    loss_config = dict(training_config.get("loss") or {})
    name = str(loss_config.get("name", "cross_entropy")).lower()
    label_smoothing = float(training_config.get("label_smoothing", 0.0))
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
    if name == "focal":
        if label_smoothing != 0.0:
            raise ValueError("Focal loss currently requires training.label_smoothing=0.0.")
        return FocalCrossEntropyLoss(
            gamma=float(loss_config.get("gamma", 2.0)),
            weight=class_weights,
        )
    raise ValueError(f"Unsupported classification loss: {name}")
