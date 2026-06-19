# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .mdp.observations import get_object_position, get_placement_target_position
from .object_in_bowl_env_cfg import (
    BOWL_SUCCESS_MAX_ANGULAR_SPEED,
    BOWL_SUCCESS_MAX_HEIGHT,
    BOWL_SUCCESS_MAX_SPEED,
    BOWL_SUCCESS_MIN_GRIPPER_OPEN,
    BOWL_SUCCESS_MIN_HEIGHT,
    BOWL_SUCCESS_RADIUS,
    PLACEMENT_TARGET_POSITION,
    PLACEMENT_TARGET_RADIUS,
)


class ObjectInBowlEnv(ManagerBasedRLEnv):
    """Manager-based environment with bowl-placement diagnostics."""

    def _reset_idx(self, env_ids: Sequence[int]):
        """Log why ended episodes did or did not satisfy bowl success."""
        if len(env_ids) == 0 or self.common_step_counter == 0:
            super()._reset_idx(env_ids)
            return

        diagnostics = self._compute_bowl_diagnostics(env_ids)
        super()._reset_idx(env_ids)
        self.extras["log"].update(diagnostics)

    def _compute_bowl_diagnostics(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Compute episode-end success-condition rates before reset changes the scene."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        robot_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
        robot_cfg.resolve(self.scene)

        robot: Articulation = self.scene[robot_cfg.name]
        object_asset: RigidObject = self.scene["object"]
        object_position = get_object_position(self)[env_ids]
        target = get_placement_target_position(self, PLACEMENT_TARGET_POSITION)[env_ids]
        xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
        height = object_position[:, 2]
        object_speed = torch.linalg.norm(object_asset.data.root_lin_vel_w[env_ids, :3], dim=1)
        object_angular_speed = torch.linalg.norm(object_asset.data.root_ang_vel_w[env_ids, :3], dim=1)
        finger_joint_pos = robot.data.joint_pos[env_ids][:, robot_cfg.joint_ids]

        is_inside_radius = xy_distance < BOWL_SUCCESS_RADIUS
        is_inside_height = torch.logical_and(height > BOWL_SUCCESS_MIN_HEIGHT, height < BOWL_SUCCESS_MAX_HEIGHT)
        is_slow = object_speed < BOWL_SUCCESS_MAX_SPEED
        is_not_spinning = object_angular_speed < BOWL_SUCCESS_MAX_ANGULAR_SPEED
        is_gripper_open = torch.all(finger_joint_pos > BOWL_SUCCESS_MIN_GRIPPER_OPEN, dim=1)
        is_success = is_inside_radius & is_inside_height & is_slow & is_not_spinning & is_gripper_open
        is_over_bridge_area = xy_distance < PLACEMENT_TARGET_RADIUS

        height_target = (BOWL_SUCCESS_MIN_HEIGHT + BOWL_SUCCESS_MAX_HEIGHT) / 2.0
        height_error = torch.abs(height - height_target)

        return {
            "Episode_Bowl/inside_radius_rate": is_inside_radius.float().mean(),
            "Episode_Bowl/inside_height_rate": is_inside_height.float().mean(),
            "Episode_Bowl/slow_linear_rate": is_slow.float().mean(),
            "Episode_Bowl/slow_angular_rate": is_not_spinning.float().mean(),
            "Episode_Bowl/gripper_open_rate": is_gripper_open.float().mean(),
            "Episode_Bowl/success_rate": is_success.float().mean(),
            "Episode_Bowl/over_bridge_area_rate": is_over_bridge_area.float().mean(),
            "Episode_Bowl/mean_xy_distance": xy_distance.mean(),
            "Episode_Bowl/mean_height_error": height_error.mean(),
        }
