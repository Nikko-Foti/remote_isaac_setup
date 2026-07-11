"""Pure helpers for validating episode-level evaluation logs."""

from __future__ import annotations

import math


def get_new_episode_log_weight(done_count: int, reset_count: float | None) -> float:
    """Return the number of newly completed episodes represented by this step's log."""
    if done_count <= 0:
        return 0.0
    if reset_count is not None and not math.isclose(reset_count, float(done_count)):
        raise ValueError(f"episode reset count {reset_count} does not match done count {done_count}")
    return float(done_count)
