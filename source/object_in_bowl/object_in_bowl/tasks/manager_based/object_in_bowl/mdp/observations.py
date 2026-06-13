# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import FrameTransformer


def ee_position(env: ManagerBasedRLEnv, sensor_name: str = "ee_frame", target_frame_index: int = 0) -> torch.Tensor:
    """End-effector position in each environment's local frame."""
    sensor: FrameTransformer = env.scene[sensor_name]
    return sensor.data.target_pos_w[:, target_frame_index, :] - env.scene.env_origins


def placement_target_position(env: ManagerBasedRLEnv, target_position: tuple[float, float, float]) -> torch.Tensor:
    """Placement target position repeated once per environment."""
    target = env.scene.env_origins.new_tensor(target_position)
    return target.repeat(env.scene.env_origins.shape[0], 1)
