"""Force one settled placement and verify the terminal reward fires exactly once."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import object_in_bowl  # noqa: F401
from object_in_bowl.env_cfg import PLACEMENT_TARGET_POSITION


def main() -> None:
    """Place the cube in the bowl and require one strict success bonus."""
    task = "Isaac-Object-In-Bowl-Franka-v0"
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=1)
    env = gym.make(task, cfg=env_cfg)
    env.reset(seed=42)
    unwrapped = env.unwrapped
    object_asset = unwrapped.scene["object"]

    pose = object_asset.data.root_pose_w.clone()
    pose[:, :3] = unwrapped.scene.env_origins + torch.tensor(
        PLACEMENT_TARGET_POSITION, device=unwrapped.device
    )
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=unwrapped.device)
    object_asset.write_root_pose_to_sim(pose)
    object_asset.write_root_velocity_to_sim(torch.zeros_like(object_asset.data.root_vel_w))

    actions = torch.zeros(env.action_space.shape, device=unwrapped.device)
    actions[:, -1] = 1.0
    final_log = None
    for _ in range(unwrapped.max_episode_length):
        _, _, terminated, truncated, info = env.step(actions)
        if torch.any(terminated | truncated):
            final_log = info.get("log", {})
            break

    env.close()
    if final_log is None:
        raise RuntimeError("forced placement did not terminate")

    expected = {
        "Episode_End/success_count": 1.0,
        "Episode_Experiment/terminal_bonus_award_count": 1.0,
        "Episode_Experiment/success_with_bad_bonus_count": 0.0,
        "Episode_Experiment/non_success_with_bonus_count": 0.0,
    }
    for key, expected_value in expected.items():
        actual = float(final_log[key])
        if actual != expected_value:
            raise RuntimeError(f"{key}={actual}, expected {expected_value}")
    print("[INFO] Forced placement earned exactly one strict terminal bonus.")


if __name__ == "__main__":
    main()
    simulation_app.close()
