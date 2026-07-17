"""Verify that centered lowering does not reduce the aggregate shaped reward."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import object_in_bowl  # noqa: F401
import torch
from object_in_bowl import rewards
from object_in_bowl.env_cfg import (
    BOWL_LOWERING_TARGET_HEIGHT,
    BOWL_SUCCESS_MIN_HEIGHT,
    CENTERED_LOWERING_REWARD_STD,
    CENTERED_LOWERING_REWARD_WEIGHT,
    LIFT_PROGRESS_REWARD_WEIGHT,
    OBJECT_LIFTED_HEIGHT,
    OBJECT_START_POSITION,
    OBJECT_TO_BOWL_XY_REWARD_STD,
    OBJECT_TO_BOWL_XY_REWARD_WEIGHT,
    PLACEMENT_TARGET_POSITION,
    PLACEMENT_TARGET_RADIUS,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """Force centered cube heights and require a nondecreasing aggregate reward while lowering."""
    task = "Isaac-Object-In-Bowl-Franka-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    env.reset(seed=42)
    unwrapped = env.unwrapped
    object_asset = unwrapped.scene["object"]

    aggregates = []
    for height in (0.106, 0.105, 0.095, 0.049):
        pose = object_asset.data.root_pose_w.clone()
        pose[:, :3] = unwrapped.scene.env_origins + torch.tensor(
            (*PLACEMENT_TARGET_POSITION[:2], height), device=unwrapped.device
        )
        pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=unwrapped.device)
        object_asset.write_root_pose_to_sim(pose)
        object_asset.write_root_velocity_to_sim(torch.zeros_like(object_asset.data.root_vel_w))

        lift = rewards.compute_object_height_progress_reward(
            unwrapped, OBJECT_START_POSITION[2], OBJECT_LIFTED_HEIGHT
        )
        transport = rewards.compute_saturated_object_to_target_xy_reward(
            unwrapped,
            PLACEMENT_TARGET_POSITION,
            OBJECT_TO_BOWL_XY_REWARD_STD,
            PLACEMENT_TARGET_RADIUS,
            OBJECT_LIFTED_HEIGHT,
        )
        lowering = rewards.compute_centered_lowering_handoff_reward(
            unwrapped,
            PLACEMENT_TARGET_POSITION,
            PLACEMENT_TARGET_RADIUS,
            BOWL_LOWERING_TARGET_HEIGHT,
            CENTERED_LOWERING_REWARD_STD,
            BOWL_SUCCESS_MIN_HEIGHT,
            OBJECT_LIFTED_HEIGHT,
            OBJECT_TO_BOWL_XY_REWARD_WEIGHT / CENTERED_LOWERING_REWARD_WEIGHT,
        )
        aggregate = (
            lift * LIFT_PROGRESS_REWARD_WEIGHT
            + transport * OBJECT_TO_BOWL_XY_REWARD_WEIGHT
            + lowering * CENTERED_LOWERING_REWARD_WEIGHT
        )
        aggregates.append(float(aggregate.item()))
        print(f"[INFO] z={height:.3f} aggregate_before_dt={aggregates[-1]:.4f}", flush=True)

    env.close()
    for higher, lower in zip(aggregates, aggregates[1:]):
        if lower + 1e-4 < higher:
            raise RuntimeError(f"aggregate reward fell while lowering: {higher:.4f} -> {lower:.4f}")
    print("[INFO] Centered lowering aggregate reward is continuous and nondecreasing.", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
