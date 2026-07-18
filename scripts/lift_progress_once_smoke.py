"""Verify that episode-best lift progress pays once and cannot be farmed by relifting."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from object_in_bowl import rewards
from object_in_bowl.env_cfg import LIFT_PROGRESS_NEAR_OBJECT_DISTANCE, OBJECT_LIFTED_HEIGHT, OBJECT_START_POSITION


def main() -> None:
    """Check new progress, holding, lowering, relifting, capping, gating, scaling, and reset behavior."""
    best = torch.zeros(1, device=args_cli.device)
    increments = []
    for progress in (0.0, 0.2, 0.2, 0.1, 0.6, 1.0, 1.0):
        increment, best = rewards.compute_episode_best_progress_increment(
            torch.tensor([progress], device=args_cli.device), best
        )
        increments.append(float(increment.item()))

    expected = (0.0, 0.2, 0.0, 0.0, 0.4, 0.4, 0.0)
    if not all(abs(actual - wanted) < 1e-5 for actual, wanted in zip(increments, expected)):
        raise RuntimeError(f"unexpected progress increments: {increments}")
    if abs(sum(increments) - 1.0) > 1e-5:
        raise RuntimeError(f"full lift should pay exactly one progress unit: {sum(increments)}")

    class FakeActionManager:
        def __init__(self, raw_actions: torch.Tensor):
            self.raw_actions = raw_actions

        def get_term(self, name: str) -> SimpleNamespace:
            if name != "gripper_action":
                raise KeyError(name)
            return SimpleNamespace(raw_actions=self.raw_actions)

    num_envs = 2
    fake_env = SimpleNamespace(
        num_envs=num_envs,
        device=args_cli.device,
        step_dt=0.02,
        ee_position=torch.zeros((num_envs, 3), device=args_cli.device),
        object_position=torch.zeros((num_envs, 3), device=args_cli.device),
        _lift_reward_best_progress=torch.zeros(num_envs, device=args_cli.device),
        action_manager=FakeActionManager(-torch.ones((num_envs, 1), device=args_cli.device)),
    )

    original_get_ee_position = rewards.get_ee_position
    original_get_object_position = rewards.get_object_position
    rewards.get_ee_position = lambda env: env.ee_position
    rewards.get_object_position = lambda env, object_cfg=None: env.object_position

    def set_progress(progress: tuple[float, float]) -> None:
        heights = torch.tensor(progress, device=args_cli.device) * (
            OBJECT_LIFTED_HEIGHT - OBJECT_START_POSITION[2]
        ) + OBJECT_START_POSITION[2]
        fake_env.object_position[:, 2] = heights
        fake_env.ee_position.copy_(fake_env.object_position)

    def production_reward() -> torch.Tensor:
        return rewards.compute_gated_episode_best_object_height_progress_reward(
            fake_env,
            initial_height=OBJECT_START_POSITION[2],
            target_height=OBJECT_LIFTED_HEIGHT,
            near_distance=LIFT_PROGRESS_NEAR_OBJECT_DISTANCE,
        )

    try:
        set_progress((0.2, 0.5))
        weighted_reward = production_reward() * 80.0 * fake_env.step_dt
        if not torch.allclose(weighted_reward, torch.tensor([16.0, 40.0], device=args_cli.device)):
            raise RuntimeError(f"multiple environments did not progress independently: {weighted_reward}")

        set_progress((0.2, 0.4))
        if torch.any(production_reward() != 0.0):
            raise RuntimeError("holding or lowering repaid lift progress")

        set_progress((0.6, 0.7))
        fake_env.ee_position[0, 0] += LIFT_PROGRESS_NEAR_OBJECT_DISTANCE + 0.01
        gated_reward = production_reward()
        if gated_reward[0] != 0.0 or gated_reward[1] <= 0.0:
            raise RuntimeError(f"inactive gate did not suppress only its environment: {gated_reward}")
        fake_env.ee_position.copy_(fake_env.object_position)
        if production_reward()[0] != 0.0:
            raise RuntimeError("progress first reached with the gate inactive paid later")

        fake_env._lift_reward_best_progress[0] = 0.0
        set_progress((0.3, 0.8))
        partial_reset_reward = production_reward() * 80.0 * fake_env.step_dt
        expected_partial = torch.tensor([24.0, 8.0], device=args_cli.device)
        if not torch.allclose(partial_reset_reward, expected_partial, atol=1e-5):
            raise RuntimeError(f"partial reset changed the wrong environment: {partial_reset_reward}")

        fake_env._lift_reward_best_progress.zero_()
        set_progress((1.5, 1.0))
        full_lift_reward = production_reward() * 80.0 * fake_env.step_dt
        if not torch.allclose(full_lift_reward, torch.tensor([80.0, 80.0], device=args_cli.device)):
            raise RuntimeError(f"full lift was not capped and scaled to 80: {full_lift_reward}")
        if not torch.all(fake_env._lift_reward_best_progress == 1.0):
            raise RuntimeError("production wrapper did not update the capped high-water mark")
    finally:
        rewards.get_ee_position = original_get_ee_position
        rewards.get_object_position = original_get_object_position

    print("[INFO] Episode-best lift progress smoke passed.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
