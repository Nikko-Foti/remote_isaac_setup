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

from .mdp.observations import get_ee_position, get_object_position, get_placement_target_position
from .object_in_bowl_env_cfg import (
    BOWL_LOWERING_RADIUS,
    BOWL_LOWERING_TARGET_HEIGHT,
    BOWL_SUCCESS_MAX_ANGULAR_SPEED,
    BOWL_SUCCESS_MAX_HEIGHT,
    BOWL_SUCCESS_MAX_SPEED,
    BOWL_SUCCESS_MIN_GRIPPER_OPEN,
    BOWL_SUCCESS_MIN_HEIGHT,
    BOWL_SUCCESS_RADIUS,
    OBJECT_LIFTED_HEIGHT,
    PLACEMENT_TARGET_POSITION,
    PLACEMENT_TARGET_RADIUS,
)

DIAGNOSTIC_NEAR_OBJECT_DISTANCE = 0.08


class ObjectInBowlEnv(ManagerBasedRLEnv):
    """Manager-based environment with bowl-placement diagnostics."""

    def __init__(self, *args, **kwargs):
        """Create episode-level buffers used only for training diagnostics."""
        super().__init__(*args, **kwargs)
        self._finger_robot_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
        self._finger_robot_cfg.resolve(self.scene)
        self._init_episode_diagnostic_buffers()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Log why ended episodes did or did not satisfy bowl success."""
        if len(env_ids) == 0 or self.common_step_counter == 0:
            super()._reset_idx(env_ids)
            self._reset_episode_diagnostics(env_ids)
            return

        diagnostics = self._compute_bowl_diagnostics(env_ids)
        diagnostics.update(self._compute_episode_diagnostics(env_ids))
        super()._reset_idx(env_ids)
        self._reset_episode_diagnostics(env_ids)
        self.extras["log"].update(diagnostics)

    def update_episode_diagnostics(self):
        """Track max/min signals that answer whether the robot ever grasped or lifted."""
        robot: Articulation = self.scene[self._finger_robot_cfg.name]
        object_position = get_object_position(self)
        ee_position = get_ee_position(self)
        finger_joint_pos = robot.data.joint_pos[:, self._finger_robot_cfg.joint_ids]

        object_z = object_position[:, 2]
        object_z_delta = object_z - self._episode_start_object_z
        ee_object_distance = torch.linalg.norm(ee_position - object_position, dim=1)
        gripper_opening = finger_joint_pos.sum(dim=1)
        min_finger_joint_pos = torch.min(finger_joint_pos, dim=1).values
        lift_range = OBJECT_LIFTED_HEIGHT - self._episode_start_object_z
        lift_progress = torch.clamp(object_z_delta / lift_range, min=0.0, max=1.0)

        gripper_action = self.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
        closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0)
        is_close_command = closing_gripper > 0.0
        is_near_object = ee_object_distance < DIAGNOSTIC_NEAR_OBJECT_DISTANCE

        self._episode_step_count += 1.0
        self._episode_close_command_count += is_close_command.float()
        self._episode_close_near_object_count += (is_close_command & is_near_object).float()
        self._episode_close_command_hit = torch.maximum(
            self._episode_close_command_hit, is_close_command.float()
        )
        self._episode_close_near_object_hit = torch.maximum(
            self._episode_close_near_object_hit, (is_close_command & is_near_object).float()
        )
        self._episode_max_object_z = torch.maximum(self._episode_max_object_z, object_z)
        self._episode_max_object_z_delta = torch.maximum(self._episode_max_object_z_delta, object_z_delta)
        self._episode_max_lift_progress = torch.maximum(self._episode_max_lift_progress, lift_progress)
        self._episode_max_lift_progress_after_close_near_object = torch.where(
            self._episode_close_near_object_hit > 0.0,
            torch.maximum(self._episode_max_lift_progress_after_close_near_object, lift_progress),
            self._episode_max_lift_progress_after_close_near_object,
        )
        self._episode_min_ee_object_distance = torch.minimum(
            self._episode_min_ee_object_distance, ee_object_distance
        )
        self._episode_min_gripper_opening = torch.minimum(self._episode_min_gripper_opening, gripper_opening)
        self._episode_min_finger_joint_pos = torch.minimum(self._episode_min_finger_joint_pos, min_finger_joint_pos)

    def _compute_bowl_diagnostics(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Compute episode-end success-condition rates before reset changes the scene."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        robot: Articulation = self.scene[self._finger_robot_cfg.name]
        object_asset: RigidObject = self.scene["object"]
        object_position = get_object_position(self)[env_ids]
        ee_position = get_ee_position(self)[env_ids]
        target = get_placement_target_position(self, PLACEMENT_TARGET_POSITION)[env_ids]
        xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
        height = object_position[:, 2]
        object_speed = torch.linalg.norm(object_asset.data.root_lin_vel_w[env_ids, :3], dim=1)
        object_angular_speed = torch.linalg.norm(object_asset.data.root_ang_vel_w[env_ids, :3], dim=1)
        finger_joint_pos = robot.data.joint_pos[env_ids][:, self._finger_robot_cfg.joint_ids]

        is_inside_radius = xy_distance < BOWL_SUCCESS_RADIUS
        is_inside_height = torch.logical_and(height > BOWL_SUCCESS_MIN_HEIGHT, height < BOWL_SUCCESS_MAX_HEIGHT)
        is_slow = object_speed < BOWL_SUCCESS_MAX_SPEED
        is_not_spinning = object_angular_speed < BOWL_SUCCESS_MAX_ANGULAR_SPEED
        is_gripper_open = torch.all(finger_joint_pos > BOWL_SUCCESS_MIN_GRIPPER_OPEN, dim=1)
        is_success = is_inside_radius & is_inside_height & is_slow & is_not_spinning & is_gripper_open
        is_over_tight_placement_area = xy_distance < PLACEMENT_TARGET_RADIUS
        is_over_lowering_area = xy_distance < BOWL_LOWERING_RADIUS

        signed_height_error = height - BOWL_LOWERING_TARGET_HEIGHT
        height_error = torch.abs(signed_height_error)
        gripper_opening = finger_joint_pos.sum(dim=1)
        min_finger_joint_pos = torch.min(finger_joint_pos, dim=1).values

        return {
            "Episode_Bowl/inside_radius_rate": is_inside_radius.float().mean(),
            "Episode_Bowl/inside_height_rate": is_inside_height.float().mean(),
            "Episode_Bowl/slow_linear_rate": is_slow.float().mean(),
            "Episode_Bowl/slow_angular_rate": is_not_spinning.float().mean(),
            "Episode_Bowl/gripper_open_rate": is_gripper_open.float().mean(),
            "Episode_Bowl/success_rate": is_success.float().mean(),
            "Episode_Bowl/over_bridge_area_rate": is_over_tight_placement_area.float().mean(),
            "Episode_Bowl/tight_placement_radius_rate": is_over_tight_placement_area.float().mean(),
            "Episode_Bowl/lowering_radius_rate": is_over_lowering_area.float().mean(),
            "Episode_Bowl/mean_xy_distance": xy_distance.mean(),
            "Episode_Bowl/mean_object_z": height.mean(),
            "Episode_Bowl/mean_ee_z": ee_position[:, 2].mean(),
            "Episode_Bowl/mean_signed_height_error": signed_height_error.mean(),
            "Episode_Bowl/mean_height_error": height_error.mean(),
            "Episode_Bowl/mean_abs_height_error": height_error.mean(),
            "Episode_Bowl/mean_gripper_opening": gripper_opening.mean(),
            "Episode_Bowl/mean_min_finger_joint_pos": min_finger_joint_pos.mean(),
        }

    def _compute_episode_diagnostics(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Compute episode-extreme diagnostics for environments about to reset."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        step_count = torch.clamp(self._episode_step_count[env_ids], min=1.0)
        lift_threshold_hit = self._episode_max_object_z[env_ids] > OBJECT_LIFTED_HEIGHT
        lift_after_close_near_object_hit = self._episode_max_lift_progress_after_close_near_object[env_ids] >= 1.0
        return {
            "Episode_Diagnostics/max_object_z": self._episode_max_object_z[env_ids].mean(),
            "Episode_Diagnostics/max_object_z_delta": self._episode_max_object_z_delta[env_ids].mean(),
            "Episode_Diagnostics/max_lift_progress": self._episode_max_lift_progress[env_ids].mean(),
            "Episode_Diagnostics/lift_threshold_hit_rate": lift_threshold_hit.float().mean(),
            "Episode_Diagnostics/max_lift_progress_after_close_near_object": (
                self._episode_max_lift_progress_after_close_near_object[env_ids].mean()
            ),
            "Episode_Diagnostics/lift_after_close_near_object_hit_rate": (
                lift_after_close_near_object_hit.float().mean()
            ),
            "Episode_Diagnostics/min_ee_object_distance": self._episode_min_ee_object_distance[env_ids].mean(),
            "Episode_Diagnostics/min_gripper_opening": self._episode_min_gripper_opening[env_ids].mean(),
            "Episode_Diagnostics/min_finger_joint_pos": self._episode_min_finger_joint_pos[env_ids].mean(),
            "Episode_Diagnostics/close_command_rate": (
                self._episode_close_command_count[env_ids] / step_count
            ).mean(),
            "Episode_Diagnostics/close_command_hit_rate": self._episode_close_command_hit[env_ids].mean(),
            "Episode_Diagnostics/close_near_object_rate": (
                self._episode_close_near_object_count[env_ids] / step_count
            ).mean(),
            "Episode_Diagnostics/close_near_object_hit_rate": (
                self._episode_close_near_object_hit[env_ids].mean()
            ),
        }

    def _init_episode_diagnostic_buffers(self):
        """Allocate buffers that summarize what happened during each episode."""
        self._episode_start_object_z = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_object_z = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_max_object_z_delta = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_max_lift_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_lift_progress_after_close_near_object = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_ee_object_distance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_gripper_opening = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_finger_joint_pos = torch.full((self.num_envs,), torch.inf, device=self.device)

    def _reset_episode_diagnostics(self, env_ids: Sequence[int]):
        """Reset episode diagnostic buffers after the scene has been reset."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        object_z = get_object_position(self)[env_ids, 2]
        self._episode_start_object_z[env_ids] = object_z
        self._episode_step_count[env_ids] = 0.0
        self._episode_close_command_count[env_ids] = 0.0
        self._episode_close_near_object_count[env_ids] = 0.0
        self._episode_close_command_hit[env_ids] = 0.0
        self._episode_close_near_object_hit[env_ids] = 0.0
        self._episode_max_object_z[env_ids] = object_z
        self._episode_max_object_z_delta[env_ids] = 0.0
        self._episode_max_lift_progress[env_ids] = 0.0
        self._episode_max_lift_progress_after_close_near_object[env_ids] = 0.0
        self._episode_min_ee_object_distance[env_ids] = torch.inf
        self._episode_min_gripper_opening[env_ids] = torch.inf
        self._episode_min_finger_joint_pos[env_ids] = torch.inf
