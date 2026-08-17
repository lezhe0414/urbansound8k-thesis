from __future__ import annotations

from collections.abc import Mapping

import torch


def clone_state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def average_state_dicts(
    state_dicts: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("At least one state dict is required for checkpoint averaging.")
    reference_keys = tuple(state_dicts[0].keys())
    if any(tuple(state_dict.keys()) != reference_keys for state_dict in state_dicts[1:]):
        raise ValueError("All checkpoint state dicts must contain identical ordered keys.")

    averaged: dict[str, torch.Tensor] = {}
    for name in reference_keys:
        values = [state_dict[name] for state_dict in state_dicts]
        reference = values[0]
        if any(value.shape != reference.shape or value.dtype != reference.dtype for value in values[1:]):
            raise ValueError(f"Checkpoint tensor mismatch for {name}.")
        if reference.is_floating_point():
            stacked = torch.stack([value.to(dtype=torch.float32) for value in values], dim=0)
            averaged[name] = stacked.mean(dim=0).to(dtype=reference.dtype)
        elif reference.is_complex():
            stacked = torch.stack([value.to(dtype=torch.complex64) for value in values], dim=0)
            averaged[name] = stacked.mean(dim=0).to(dtype=reference.dtype)
        else:
            averaged[name] = reference.clone()
    return averaged
