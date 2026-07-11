"""Pure helpers for validating episode-level evaluation logs."""

from __future__ import annotations


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
