from __future__ import annotations

from collections.abc import Sequence

import torch


def compute_lifted_object_inside_target_radius(
    object_position: torch.Tensor,
    target: torch.Tensor,
    radius: float,
    minimal_height: float,
) -> torch.Tensor:
    """Compute the shared lift-and-radius predicate from position tensors."""
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    return (object_position[:, 2] > minimal_height) & (xy_distance <= radius)


def update_first_event_latch(active: torch.Tensor, has_fired: torch.Tensor) -> torch.Tensor:
    """Latch active environments and return only their first active step."""
    first_event = active & torch.logical_not(has_fired)
    has_fired.logical_or_(active)
    return first_event


def reset_event_latch(has_fired: torch.Tensor, env_ids: Sequence[int] | None = None) -> None:
    """Reset all or selected environments in an event latch."""
    if env_ids is None:
        env_ids = slice(None)
    has_fired[env_ids] = False
