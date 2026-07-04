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
    LIFT_PROGRESS_NEAR_OBJECT_DISTANCE,
    OBJECT_LIFTED_HEIGHT,
    OBJECT_START_POSITION,
    PLACEMENT_TARGET_POSITION,
    PLACEMENT_TARGET_RADIUS,
)


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

        arm_action = self.action_manager.get_term("arm_action").raw_actions
        gripper_action = self.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
        closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0)
        is_close_command = closing_gripper > 0.0
        is_near_object = ee_object_distance < LIFT_PROGRESS_NEAR_OBJECT_DISTANCE
        is_gate_active = is_close_command & is_near_object
        current_step = self._episode_step_count + 1.0

        self._episode_step_count += 1.0
        self._episode_close_command_count += is_close_command.float()
        self._episode_close_near_object_count += is_gate_active.float()
        self._episode_close_command_hit = torch.maximum(
            self._episode_close_command_hit, is_close_command.float()
        )
        self._episode_close_near_object_hit = torch.maximum(
            self._episode_close_near_object_hit, is_gate_active.float()
        )

        first_close_near = is_gate_active & (self._episode_first_close_near_step < 0.0)
        self._episode_first_close_near_step = torch.where(
            first_close_near, current_step, self._episode_first_close_near_step
        )

        new_max_lift_progress = lift_progress > self._episode_max_lift_progress
        self._episode_gate_active_at_max_lift_progress = torch.where(
            new_max_lift_progress, is_gate_active.float(), self._episode_gate_active_at_max_lift_progress
        )
        self._episode_near_object_at_max_lift_progress = torch.where(
            new_max_lift_progress, is_near_object.float(), self._episode_near_object_at_max_lift_progress
        )
        self._episode_close_command_at_max_lift_progress = torch.where(
            new_max_lift_progress, is_close_command.float(), self._episode_close_command_at_max_lift_progress
        )
        self._episode_max_object_z = torch.maximum(self._episode_max_object_z, object_z)
        self._episode_max_object_z_delta = torch.maximum(self._episode_max_object_z_delta, object_z_delta)
        self._episode_max_lift_progress = torch.maximum(self._episode_max_lift_progress, lift_progress)
        self._episode_max_lift_progress_while_gate_active = torch.where(
            is_gate_active,
            torch.maximum(self._episode_max_lift_progress_while_gate_active, lift_progress),
            self._episode_max_lift_progress_while_gate_active,
        )
        self._episode_max_lift_progress_after_close_near_object = torch.where(
            self._episode_close_near_object_hit > 0.0,
            torch.maximum(self._episode_max_lift_progress_after_close_near_object, lift_progress),
            self._episode_max_lift_progress_after_close_near_object,
        )

        crossed_lift_threshold = object_z > OBJECT_LIFTED_HEIGHT
        first_lift_threshold_crossing = crossed_lift_threshold & (self._episode_lift_threshold_hit == 0.0)
        self._episode_gate_active_at_first_lift_threshold_crossing = torch.where(
            first_lift_threshold_crossing,
            is_gate_active.float(),
            self._episode_gate_active_at_first_lift_threshold_crossing,
        )
        self._episode_first_lift_threshold_step = torch.where(
            first_lift_threshold_crossing, current_step, self._episode_first_lift_threshold_step
        )
        self._episode_lift_threshold_hit = torch.maximum(
            self._episode_lift_threshold_hit, crossed_lift_threshold.float()
        )

        crossed_lift_005m = object_z_delta > 0.005
        crossed_lift_020m = object_z_delta > 0.020
        post_lift_005m = (self._episode_lift_005m_hit > 0.0) | crossed_lift_005m
        post_lift_020m = (self._episode_lift_020m_hit > 0.0) | crossed_lift_020m
        self._episode_lift_005m_hit = torch.maximum(self._episode_lift_005m_hit, crossed_lift_005m.float())
        self._episode_lift_020m_hit = torch.maximum(self._episode_lift_020m_hit, crossed_lift_020m.float())
        self._episode_post_lift_005m_step_count += post_lift_005m.float()
        self._episode_post_lift_005m_gate_count += (post_lift_005m & is_gate_active).float()
        self._episode_post_lift_005m_near_count += (post_lift_005m & is_near_object).float()
        self._episode_post_lift_005m_close_count += (post_lift_005m & is_close_command).float()
        self._episode_post_lift_005m_gate_off_far_count += (
            post_lift_005m & torch.logical_not(is_gate_active) & torch.logical_not(is_near_object)
        ).float()
        self._episode_post_lift_005m_gate_off_open_count += (
            post_lift_005m & torch.logical_not(is_gate_active) & torch.logical_not(is_close_command)
        ).float()
        self._episode_post_lift_020m_step_count += post_lift_020m.float()
        self._episode_post_lift_020m_gate_count += (post_lift_020m & is_gate_active).float()
        self._episode_post_lift_020m_near_count += (post_lift_020m & is_near_object).float()
        self._episode_post_lift_020m_close_count += (post_lift_020m & is_close_command).float()
        self._episode_post_lift_020m_gate_off_far_count += (
            post_lift_020m & torch.logical_not(is_gate_active) & torch.logical_not(is_near_object)
        ).float()
        self._episode_post_lift_020m_gate_off_open_count += (
            post_lift_020m & torch.logical_not(is_gate_active) & torch.logical_not(is_close_command)
        ).float()

        has_previous_action = self._episode_has_previous_action > 0.0
        arm_action_delta = torch.linalg.norm(arm_action - self._episode_previous_arm_action, dim=1)
        gripper_switched = is_close_command != self._episode_previous_gripper_close_command
        self._episode_arm_action_delta_sq_sum += torch.where(
            has_previous_action, arm_action_delta.square(), torch.zeros_like(arm_action_delta)
        )
        self._episode_action_delta_count += has_previous_action.float()
        self._episode_gripper_switch_count += (has_previous_action & gripper_switched).float()
        self._episode_previous_arm_action = arm_action.clone()
        self._episode_previous_gripper_close_command = is_close_command.clone()
        self._episode_has_previous_action = torch.ones_like(self._episode_has_previous_action)

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
        max_object_z_delta = self._episode_max_object_z_delta[env_ids]
        lift_threshold_hit = self._episode_lift_threshold_hit[env_ids] > 0.0
        failed_to_lift = torch.logical_not(lift_threshold_hit)
        failed_partial_lift_005m = failed_to_lift & (max_object_z_delta > 0.005)
        failed_partial_lift_020m = failed_to_lift & (max_object_z_delta > 0.020)
        failed_partial_lift_005m_count = failed_partial_lift_005m.float().sum()
        failed_partial_lift_020m_count = failed_partial_lift_020m.float().sum()
        lift_after_close_near_object_hit = self._episode_max_lift_progress_after_close_near_object[env_ids] >= 1.0
        close_near_step_fraction = (self._episode_close_near_object_count[env_ids] / step_count).mean()
        start_object_xy = self._episode_start_object_xy[env_ids]
        success_start_xy = self._masked_vector_mean(start_object_xy, lift_threshold_hit)
        failure_start_xy = self._masked_vector_mean(start_object_xy, failed_to_lift)
        success_count = lift_threshold_hit.float().sum()
        failure_count = failed_to_lift.float().sum()
        first_lift_threshold_count = (self._episode_first_lift_threshold_step[env_ids] >= 0.0).float().sum()
        action_delta_count = torch.clamp(self._episode_action_delta_count[env_ids], min=1.0)
        arm_action_delta_rms = torch.sqrt(self._episode_arm_action_delta_sq_sum[env_ids] / action_delta_count)
        gripper_switch_rate = self._episode_gripper_switch_count[env_ids] / action_delta_count
        start_x_offset = start_object_xy[:, 0] - OBJECT_START_POSITION[0]
        start_y_offset = start_object_xy[:, 1] - OBJECT_START_POSITION[1]

        return {
            "Episode_Diagnostics/reset_env_count": torch.as_tensor(len(env_ids), device=self.device, dtype=torch.float32),
            "Episode_Diagnostics/max_object_z": self._episode_max_object_z[env_ids].mean(),
            "Episode_Diagnostics/max_object_z_delta": max_object_z_delta.mean(),
            "Episode_Diagnostics/max_lift_delta_mean": max_object_z_delta.mean(),
            "Episode_Diagnostics/max_lift_progress": self._episode_max_lift_progress[env_ids].mean(),
            "Episode_Diagnostics/max_lift_delta_lt_005m_rate": (max_object_z_delta < 0.005).float().mean(),
            "Episode_Diagnostics/max_lift_delta_005_010m_rate": (
                torch.logical_and(max_object_z_delta >= 0.005, max_object_z_delta < 0.010).float().mean()
            ),
            "Episode_Diagnostics/max_lift_delta_010_020m_rate": (
                torch.logical_and(max_object_z_delta >= 0.010, max_object_z_delta < 0.020).float().mean()
            ),
            "Episode_Diagnostics/max_lift_delta_020_050m_rate": (
                torch.logical_and(max_object_z_delta >= 0.020, max_object_z_delta <= 0.050).float().mean()
            ),
            "Episode_Diagnostics/max_lift_delta_gt_050m_rate": (max_object_z_delta > 0.050).float().mean(),
            "Episode_Diagnostics/lift_005m_hit_rate": (max_object_z_delta > 0.005).float().mean(),
            "Episode_Diagnostics/lift_010m_hit_rate": (max_object_z_delta > 0.010).float().mean(),
            "Episode_Diagnostics/lift_020m_hit_rate": (max_object_z_delta > 0.020).float().mean(),
            "Episode_Diagnostics/lift_threshold_hit_rate": lift_threshold_hit.float().mean(),
            "Episode_Diagnostics/max_lift_progress_after_close_near_object": (
                self._episode_max_lift_progress_after_close_near_object[env_ids].mean()
            ),
            "Episode_Diagnostics/ever_lifted_after_first_close_near_rate": (
                lift_after_close_near_object_hit.float().mean()
            ),
            "Episode_Diagnostics/max_lift_progress_while_gate_active": (
                self._episode_max_lift_progress_while_gate_active[env_ids].mean()
            ),
            "Episode_Diagnostics/close_near_object_step_fraction": close_near_step_fraction,
            "Episode_Diagnostics/ever_close_near_object_rate": self._episode_close_near_object_hit[env_ids].mean(),
            "Episode_Diagnostics/gate_active_at_max_lift_rate": (
                self._episode_gate_active_at_max_lift_progress[env_ids].mean()
            ),
            "Episode_Diagnostics/near_object_at_max_lift_rate": (
                self._episode_near_object_at_max_lift_progress[env_ids].mean()
            ),
            "Episode_Diagnostics/close_command_at_max_lift_rate": (
                self._episode_close_command_at_max_lift_progress[env_ids].mean()
            ),
            "Episode_Diagnostics/failed_partial_lift_005m_rate": failed_partial_lift_005m.float().mean(),
            "Episode_Diagnostics/failed_partial_lift_020m_rate": failed_partial_lift_020m.float().mean(),
            "Episode_Diagnostics/failed_partial_lift_005m_episode_count": failed_partial_lift_005m_count,
            "Episode_Diagnostics/failed_partial_lift_020m_episode_count": failed_partial_lift_020m_count,
            "Episode_Diagnostics/gate_active_at_max_lift_rate_failed_partial_005m": self._masked_mean(
                self._episode_gate_active_at_max_lift_progress[env_ids], failed_partial_lift_005m
            ),
            "Episode_Diagnostics/near_object_at_max_lift_rate_failed_partial_005m": self._masked_mean(
                self._episode_near_object_at_max_lift_progress[env_ids], failed_partial_lift_005m
            ),
            "Episode_Diagnostics/close_command_at_max_lift_rate_failed_partial_005m": self._masked_mean(
                self._episode_close_command_at_max_lift_progress[env_ids], failed_partial_lift_005m
            ),
            "Episode_Diagnostics/gate_active_at_max_lift_rate_failed_partial_020m": self._masked_mean(
                self._episode_gate_active_at_max_lift_progress[env_ids], failed_partial_lift_020m
            ),
            "Episode_Diagnostics/near_object_at_max_lift_rate_failed_partial_020m": self._masked_mean(
                self._episode_near_object_at_max_lift_progress[env_ids], failed_partial_lift_020m
            ),
            "Episode_Diagnostics/close_command_at_max_lift_rate_failed_partial_020m": self._masked_mean(
                self._episode_close_command_at_max_lift_progress[env_ids], failed_partial_lift_020m
            ),
            "Episode_Diagnostics/gate_rate_after_lift_005m": self._safe_rate(
                self._episode_post_lift_005m_gate_count[env_ids].sum(),
                self._episode_post_lift_005m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/post_lift_005m_step_count": self._episode_post_lift_005m_step_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_005m_gate_count": self._episode_post_lift_005m_gate_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_005m_near_count": self._episode_post_lift_005m_near_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_005m_close_count": self._episode_post_lift_005m_close_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_005m_gate_off_far_count": self._episode_post_lift_005m_gate_off_far_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_005m_gate_off_open_count": self._episode_post_lift_005m_gate_off_open_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/near_rate_after_lift_005m": self._safe_rate(
                self._episode_post_lift_005m_near_count[env_ids].sum(),
                self._episode_post_lift_005m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/close_rate_after_lift_005m": self._safe_rate(
                self._episode_post_lift_005m_close_count[env_ids].sum(),
                self._episode_post_lift_005m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_off_after_lift_005m_far_component_rate": self._safe_rate(
                self._episode_post_lift_005m_gate_off_far_count[env_ids].sum(),
                self._episode_post_lift_005m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_off_after_lift_005m_open_component_rate": self._safe_rate(
                self._episode_post_lift_005m_gate_off_open_count[env_ids].sum(),
                self._episode_post_lift_005m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_rate_after_lift_020m": self._safe_rate(
                self._episode_post_lift_020m_gate_count[env_ids].sum(),
                self._episode_post_lift_020m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/post_lift_020m_step_count": self._episode_post_lift_020m_step_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_020m_gate_count": self._episode_post_lift_020m_gate_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_020m_near_count": self._episode_post_lift_020m_near_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_020m_close_count": self._episode_post_lift_020m_close_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_020m_gate_off_far_count": self._episode_post_lift_020m_gate_off_far_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/post_lift_020m_gate_off_open_count": self._episode_post_lift_020m_gate_off_open_count[
                env_ids
            ].sum(),
            "Episode_Diagnostics/near_rate_after_lift_020m": self._safe_rate(
                self._episode_post_lift_020m_near_count[env_ids].sum(),
                self._episode_post_lift_020m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/close_rate_after_lift_020m": self._safe_rate(
                self._episode_post_lift_020m_close_count[env_ids].sum(),
                self._episode_post_lift_020m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_off_after_lift_020m_far_component_rate": self._safe_rate(
                self._episode_post_lift_020m_gate_off_far_count[env_ids].sum(),
                self._episode_post_lift_020m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_off_after_lift_020m_open_component_rate": self._safe_rate(
                self._episode_post_lift_020m_gate_off_open_count[env_ids].sum(),
                self._episode_post_lift_020m_step_count[env_ids].sum(),
            ),
            "Episode_Diagnostics/gate_active_at_first_lift_threshold_crossing_rate": self._masked_mean(
                self._episode_gate_active_at_first_lift_threshold_crossing[env_ids],
                self._episode_first_lift_threshold_step[env_ids] >= 0.0,
            ),
            "Episode_Diagnostics/close_near_before_lift_threshold_rate": self._masked_mean(
                (
                    (self._episode_first_close_near_step[env_ids] >= 0.0)
                    & (
                        self._episode_first_close_near_step[env_ids]
                        <= self._episode_first_lift_threshold_step[env_ids]
                    )
                ).float(),
                self._episode_first_lift_threshold_step[env_ids] >= 0.0,
            ),
            "Episode_Diagnostics/first_lift_threshold_crossing_episode_count": first_lift_threshold_count,
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
            "Episode_Diagnostics/start_object_x_mean_success": success_start_xy[0],
            "Episode_Diagnostics/start_object_y_mean_success": success_start_xy[1],
            "Episode_Diagnostics/start_object_x_mean_failure": failure_start_xy[0],
            "Episode_Diagnostics/start_object_y_mean_failure": failure_start_xy[1],
            "Episode_Diagnostics/success_episode_count": success_count,
            "Episode_Diagnostics/failure_episode_count": failure_count,
            "Episode_Diagnostics/failure_start_x_negative_offset_rate": self._masked_mean(
                (start_x_offset < 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/failure_start_x_positive_offset_rate": self._masked_mean(
                (start_x_offset >= 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/failure_start_y_negative_offset_rate": self._masked_mean(
                (start_y_offset < 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/failure_start_y_positive_offset_rate": self._masked_mean(
                (start_y_offset >= 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/arm_action_delta_rms": arm_action_delta_rms.mean(),
            "Episode_Diagnostics/arm_action_delta_rms_success": self._masked_mean(
                arm_action_delta_rms, lift_threshold_hit
            ),
            "Episode_Diagnostics/arm_action_delta_rms_failure": self._masked_mean(
                arm_action_delta_rms, failed_to_lift
            ),
            "Episode_Diagnostics/gripper_switch_rate": gripper_switch_rate.mean(),
            "Episode_Diagnostics/gripper_switch_rate_success": self._masked_mean(
                gripper_switch_rate, lift_threshold_hit
            ),
            "Episode_Diagnostics/gripper_switch_rate_failure": self._masked_mean(
                gripper_switch_rate, failed_to_lift
            ),
        }

    def _init_episode_diagnostic_buffers(self):
        """Allocate buffers that summarize what happened during each episode."""
        self._episode_start_object_z = torch.zeros(self.num_envs, device=self.device)
        self._episode_start_object_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_object_z = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_max_object_z_delta = torch.full((self.num_envs,), -torch.inf, device=self.device)
        self._episode_max_lift_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_lift_progress_after_close_near_object = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_lift_progress_while_gate_active = torch.zeros(self.num_envs, device=self.device)
        self._episode_gate_active_at_max_lift_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_near_object_at_max_lift_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_at_max_lift_progress = torch.zeros(self.num_envs, device=self.device)
        self._episode_first_close_near_step = torch.full((self.num_envs,), -1.0, device=self.device)
        self._episode_first_lift_threshold_step = torch.full((self.num_envs,), -1.0, device=self.device)
        self._episode_gate_active_at_first_lift_threshold_crossing = torch.zeros(self.num_envs, device=self.device)
        self._episode_lift_005m_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_lift_020m_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_lift_threshold_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_gate_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_near_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_close_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_gate_off_far_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_005m_gate_off_open_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_gate_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_near_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_close_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_gate_off_far_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_post_lift_020m_gate_off_open_count = torch.zeros(self.num_envs, device=self.device)
        arm_action = self.action_manager.get_term("arm_action").raw_actions
        self._episode_previous_arm_action = torch.zeros_like(arm_action)
        self._episode_previous_gripper_close_command = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_has_previous_action = torch.zeros(self.num_envs, device=self.device)
        self._episode_arm_action_delta_sq_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_delta_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_gripper_switch_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_ee_object_distance = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_gripper_opening = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_min_finger_joint_pos = torch.full((self.num_envs,), torch.inf, device=self.device)

    def _reset_episode_diagnostics(self, env_ids: Sequence[int]):
        """Reset episode diagnostic buffers after the scene has been reset."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        object_position = get_object_position(self)[env_ids]
        self._episode_start_object_z[env_ids] = object_position[:, 2]
        self._episode_start_object_xy[env_ids] = object_position[:, :2]
        self._episode_step_count[env_ids] = 0.0
        self._episode_close_command_count[env_ids] = 0.0
        self._episode_close_near_object_count[env_ids] = 0.0
        self._episode_close_command_hit[env_ids] = 0.0
        self._episode_close_near_object_hit[env_ids] = 0.0
        self._episode_max_object_z[env_ids] = object_position[:, 2]
        self._episode_max_object_z_delta[env_ids] = 0.0
        self._episode_max_lift_progress[env_ids] = 0.0
        self._episode_max_lift_progress_after_close_near_object[env_ids] = 0.0
        self._episode_max_lift_progress_while_gate_active[env_ids] = 0.0
        self._episode_gate_active_at_max_lift_progress[env_ids] = 0.0
        self._episode_near_object_at_max_lift_progress[env_ids] = 0.0
        self._episode_close_command_at_max_lift_progress[env_ids] = 0.0
        self._episode_first_close_near_step[env_ids] = -1.0
        self._episode_first_lift_threshold_step[env_ids] = -1.0
        self._episode_gate_active_at_first_lift_threshold_crossing[env_ids] = 0.0
        self._episode_lift_005m_hit[env_ids] = 0.0
        self._episode_lift_020m_hit[env_ids] = 0.0
        self._episode_lift_threshold_hit[env_ids] = 0.0
        self._episode_post_lift_005m_step_count[env_ids] = 0.0
        self._episode_post_lift_005m_gate_count[env_ids] = 0.0
        self._episode_post_lift_005m_near_count[env_ids] = 0.0
        self._episode_post_lift_005m_close_count[env_ids] = 0.0
        self._episode_post_lift_005m_gate_off_far_count[env_ids] = 0.0
        self._episode_post_lift_005m_gate_off_open_count[env_ids] = 0.0
        self._episode_post_lift_020m_step_count[env_ids] = 0.0
        self._episode_post_lift_020m_gate_count[env_ids] = 0.0
        self._episode_post_lift_020m_near_count[env_ids] = 0.0
        self._episode_post_lift_020m_close_count[env_ids] = 0.0
        self._episode_post_lift_020m_gate_off_far_count[env_ids] = 0.0
        self._episode_post_lift_020m_gate_off_open_count[env_ids] = 0.0
        self._episode_previous_arm_action[env_ids] = 0.0
        self._episode_previous_gripper_close_command[env_ids] = False
        self._episode_has_previous_action[env_ids] = 0.0
        self._episode_arm_action_delta_sq_sum[env_ids] = 0.0
        self._episode_action_delta_count[env_ids] = 0.0
        self._episode_gripper_switch_count[env_ids] = 0.0
        self._episode_min_ee_object_distance[env_ids] = torch.inf
        self._episode_min_gripper_opening[env_ids] = torch.inf
        self._episode_min_finger_joint_pos[env_ids] = torch.inf

    def _safe_rate(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Return numerator / denominator, or zero when no samples exist."""
        return torch.where(
            denominator > 0.0,
            numerator / torch.clamp(denominator, min=1.0),
            torch.zeros((), device=self.device),
        )

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average values only over true mask entries, returning zero if empty."""
        mask = mask.float()
        count = mask.sum()
        return self._safe_rate((values * mask).sum(), count)

    def _masked_vector_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average vector values only over true mask entries, returning zeros if empty."""
        mask = mask.float().unsqueeze(-1)
        count = mask.sum()
        return torch.where(
            count > 0.0,
            (values * mask).sum(dim=0) / torch.clamp(count, min=1.0),
            torch.zeros(values.shape[1], device=self.device),
        )
