"""Verify that episode-best lift progress pays once and cannot be farmed by relifting."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from object_in_bowl.rewards import compute_episode_best_progress_increment


def main() -> None:
    """Check new progress, holding, lowering, relifting, capping, and reset behavior."""
    best = torch.zeros(1, device=args_cli.device)
    increments = []
    for progress in (0.0, 0.2, 0.2, 0.1, 0.6, 1.0, 1.0):
        increment, best = compute_episode_best_progress_increment(
            torch.tensor([progress], device=args_cli.device), best
        )
        increments.append(float(increment.item()))

    expected = (0.0, 0.2, 0.0, 0.0, 0.4, 0.4, 0.0)
    if not all(abs(actual - wanted) < 1e-5 for actual, wanted in zip(increments, expected)):
        raise RuntimeError(f"unexpected progress increments: {increments}")
    if abs(sum(increments) - 1.0) > 1e-5:
        raise RuntimeError(f"full lift should pay exactly one progress unit: {sum(increments)}")

    reset_best = torch.zeros_like(best)
    reset_increment, _ = compute_episode_best_progress_increment(torch.tensor([0.5], device=args_cli.device), reset_best)
    if abs(float(reset_increment.item()) - 0.5) > 1e-5:
        raise RuntimeError("reset did not allow a new episode to earn lift progress")

    print("[INFO] Episode-best lift progress smoke passed.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
