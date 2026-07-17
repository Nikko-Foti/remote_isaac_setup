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


def validate_exclusive_end_counts(
    completed_episodes: int,
    success_count: float,
    timeout_count: float,
    drop_count: float,
    other_count: float,
) -> None:
    """Require every completed episode to have exactly one logged end reason."""
    end_count = success_count + timeout_count + drop_count + other_count
    if end_count != float(completed_episodes):
        raise ValueError(f"completed {completed_episodes} episodes but end-reason count is {end_count}")


def validate_sequential_funnel(stage_counts: list[tuple[str, float]]) -> None:
    """Require each sequential stage count to be no larger than its predecessor."""
    for name, count in stage_counts:
        if count < 0.0:
            raise ValueError(f"funnel stage {name} has negative count {count}")
    for (previous_name, previous_count), (name, count) in zip(stage_counts, stage_counts[1:]):
        if count > previous_count:
            raise ValueError(
                f"funnel stage {name} count {count} exceeds {previous_name} count {previous_count}"
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
