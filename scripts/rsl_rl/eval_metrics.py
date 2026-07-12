"""Pure helpers for validating episode-level evaluation logs."""

from __future__ import annotations

import math


def get_new_episode_log_weight(done_count: int, reset_count: float | None) -> float:
    """Return the number of newly completed episodes represented by this step's log."""
    if done_count <= 0:
        return 0.0
    if reset_count is not None and reset_count != float(done_count):
        raise ValueError(f"episode reset count {reset_count} does not match done count {done_count}")
    return float(done_count)


def validate_episode_log_total(completed_episodes: int, logged_episode_weight: float) -> None:
    """Require exactly one aggregated log entry per completed episode."""
    if logged_episode_weight != float(completed_episodes):
        raise ValueError(
            f"completed {completed_episodes} episodes but aggregated weight {logged_episode_weight}"
        )


def summarize_distribution(values: list[float]) -> dict[str, float]:
    """Return exact summary statistics for collected episode samples."""
    if not values:
        return {}
    ordered = sorted(values)

    def quantile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        fraction = index - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "mean": sum(ordered) / len(ordered),
        "median": quantile(0.5),
        "p10": quantile(0.1),
        "p90": quantile(0.9),
    }
