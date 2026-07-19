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

from .env_cfg import (
    BOWL_LOWERING_RADIUS,
    BOWL_LOWERING_TARGET_HEIGHT,
    BOWL_RELEASE_CONTACT_FORCE_THRESHOLD,
    BOWL_SUCCESS_MAX_ANGULAR_SPEED,
    BOWL_SUCCESS_MAX_HEIGHT,
    BOWL_SUCCESS_MAX_SPEED,
    BOWL_SUCCESS_MIN_GRIPPER_OPEN,
    BOWL_SUCCESS_MIN_HEIGHT,
    BOWL_SUCCESS_RADIUS,
    BOWL_SUPPORT_FORCE_THRESHOLD,
    LIFT_PROGRESS_NEAR_OBJECT_DISTANCE,
    OBJECT_LIFTED_HEIGHT,
    OBJECT_START_POSITION,
    PLACEMENT_TARGET_POSITION,
    PLACEMENT_TARGET_RADIUS,
    VERIFIED_GRASP_FORCE_THRESHOLD,
    VERIFIED_GRASP_HISTORY_LENGTH,
)
from .observations import get_ee_position, get_object_position, get_placement_target_position
from .rewards import check_bowl_support_contact, check_finger_object_contact, check_verified_grasp


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
        diagnostics.update(self._compute_end_reason_diagnostics(env_ids))
        diagnostics.update(self._compute_reward_sum_diagnostics(env_ids))
        successful_lift_height_samples = self._get_successful_lift_height_samples(env_ids)
        episode_samples = self._get_episode_samples(env_ids)
        super()._reset_idx(env_ids)
        self._reset_episode_diagnostics(env_ids)
        self.extras["log"].update(diagnostics)
        self.extras["successful_lift_height_samples"] = successful_lift_height_samples
        self.extras["episode_samples"] = episode_samples

    def update_success_dwell(self, is_success_state: torch.Tensor, required_steps: int) -> torch.Tensor:
        """Require the physical success state to persist before terminating an episode."""
        self._success_dwell_steps = torch.where(
            is_success_state,
            self._success_dwell_steps + 1.0,
            torch.zeros_like(self._success_dwell_steps),
        )
        self._episode_max_success_dwell_steps = torch.maximum(
            self._episode_max_success_dwell_steps, self._success_dwell_steps
        )
        self._episode_success_5_hit = torch.maximum(
            self._episode_success_5_hit, (self._success_dwell_steps >= 5).float()
        )
        self._episode_success_10_hit = torch.maximum(
            self._episode_success_10_hit, (self._success_dwell_steps >= 10).float()
        )
        return self._success_dwell_steps >= required_steps

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
        target = get_placement_target_position(self, PLACEMENT_TARGET_POSITION)
        xy_distance_to_bowl = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
        object_asset: RigidObject = self.scene["object"]
        object_speed = torch.linalg.norm(object_asset.data.root_lin_vel_w[:, :3], dim=1)
        object_angular_speed = torch.linalg.norm(object_asset.data.root_ang_vel_w[:, :3], dim=1)

        arm_action = self.action_manager.get_term("arm_action").raw_actions
        gripper_action = self.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
        closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0)
        is_close_command = closing_gripper > 0.0
        is_near_object = ee_object_distance < LIFT_PROGRESS_NEAR_OBJECT_DISTANCE
        is_gate_active = is_close_command & is_near_object
        is_verified_grasp = check_verified_grasp(
            self,
            force_threshold=VERIFIED_GRASP_FORCE_THRESHOLD,
            history_length=VERIFIED_GRASP_HISTORY_LENGTH,
        )
        is_gripper_open = torch.all(finger_joint_pos > BOWL_SUCCESS_MIN_GRIPPER_OPEN, dim=1)
        has_finger_contact = check_finger_object_contact(self, BOWL_RELEASE_CONTACT_FORCE_THRESHOLD)
        has_bowl_support = check_bowl_support_contact(
            self,
            PLACEMENT_TARGET_POSITION,
            BOWL_SUCCESS_RADIUS,
            BOWL_SUCCESS_MIN_HEIGHT,
            BOWL_SUCCESS_MAX_HEIGHT,
            BOWL_SUPPORT_FORCE_THRESHOLD,
        )
        current_step = self._episode_step_count + 1.0

        self._episode_step_count += 1.0
        self._episode_close_command_count += is_close_command.float()
        self._episode_close_near_object_count += is_gate_active.float()
        self._episode_verified_grasp_count += is_verified_grasp.float()
        self._episode_close_command_hit = torch.maximum(
            self._episode_close_command_hit, is_close_command.float()
        )
        self._episode_close_near_object_hit = torch.maximum(
            self._episode_close_near_object_hit, is_gate_active.float()
        )
        self._episode_verified_grasp_hit = torch.maximum(
            self._episode_verified_grasp_hit, is_verified_grasp.float()
        )

        verified_stage = self._episode_verified_grasp_hit > 0.0
        lifted_stage = verified_stage & (object_z > OBJECT_LIFTED_HEIGHT)
        self._episode_funnel_lift_hit = torch.maximum(self._episode_funnel_lift_hit, lifted_stage.float())
        broad_entry_stage = (self._episode_funnel_lift_hit > 0.0) & (xy_distance_to_bowl < BOWL_SUCCESS_RADIUS)
        self._episode_funnel_broad_entry_hit = torch.maximum(
            self._episode_funnel_broad_entry_hit, broad_entry_stage.float()
        )
        centered_stage = (self._episode_funnel_broad_entry_hit > 0.0) & (
            xy_distance_to_bowl < PLACEMENT_TARGET_RADIUS
        )
        self._episode_funnel_centered_hit = torch.maximum(
            self._episode_funnel_centered_hit, centered_stage.float()
        )
        lowered_stage = (self._episode_funnel_centered_hit > 0.0) & (
            (object_z > BOWL_SUCCESS_MIN_HEIGHT) & (object_z < BOWL_SUCCESS_MAX_HEIGHT)
        )
        self._episode_funnel_lowered_hit = torch.maximum(
            self._episode_funnel_lowered_hit, lowered_stage.float()
        )
        opened_stage = (self._episode_funnel_lowered_hit > 0.0) & is_gripper_open
        self._episode_funnel_opened_hit = torch.maximum(self._episode_funnel_opened_hit, opened_stage.float())
        released_stage = (
            (self._episode_funnel_opened_hit > 0.0)
            & is_gripper_open
            & torch.logical_not(has_finger_contact)
        )
        self._episode_funnel_released_hit = torch.maximum(
            self._episode_funnel_released_hit, released_stage.float()
        )
        supported_settled_stage = (
            (self._episode_funnel_released_hit > 0.0)
            & (xy_distance_to_bowl < BOWL_SUCCESS_RADIUS)
            & (object_z > BOWL_SUCCESS_MIN_HEIGHT)
            & (object_z < BOWL_SUCCESS_MAX_HEIGHT)
            & released_stage
            & has_bowl_support
            & (object_speed < BOWL_SUCCESS_MAX_SPEED)
            & (object_angular_speed < BOWL_SUCCESS_MAX_ANGULAR_SPEED)
        )
        self._episode_funnel_supported_settled_hit = torch.maximum(
            self._episode_funnel_supported_settled_hit, supported_settled_stage.float()
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
        has_lifted = self._episode_lift_threshold_hit > 0.0
        self._episode_post_lift_threshold_step_count += has_lifted.float()
        self._episode_above_lift_threshold_step_count += (has_lifted & crossed_lift_threshold).float()
        self._episode_fell_below_lift_threshold = torch.maximum(
            self._episode_fell_below_lift_threshold,
            (has_lifted & torch.logical_not(crossed_lift_threshold)).float(),
        )
        self._episode_min_xy_distance_after_lift = torch.where(
            has_lifted,
            torch.minimum(self._episode_min_xy_distance_after_lift, xy_distance_to_bowl),
            self._episode_min_xy_distance_after_lift,
        )
        self._episode_bowl_radius_hit_after_lift = torch.maximum(
            self._episode_bowl_radius_hit_after_lift,
            (has_lifted & (xy_distance_to_bowl < BOWL_SUCCESS_RADIUS)).float(),
        )
        self._episode_tight_radius_hit_after_lift = torch.maximum(
            self._episode_tight_radius_hit_after_lift,
            (has_lifted & (xy_distance_to_bowl < PLACEMENT_TARGET_RADIUS)).float(),
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
        has_finger_contact = check_finger_object_contact(self, BOWL_RELEASE_CONTACT_FORCE_THRESHOLD)[env_ids]
        has_object_contact = check_bowl_support_contact(
            self,
            PLACEMENT_TARGET_POSITION,
            BOWL_SUCCESS_RADIUS,
            BOWL_SUCCESS_MIN_HEIGHT,
            BOWL_SUCCESS_MAX_HEIGHT,
            BOWL_SUPPORT_FORCE_THRESHOLD,
        )[env_ids]
        is_released = is_gripper_open & torch.logical_not(has_finger_contact)
        has_bowl_support = is_inside_radius & is_inside_height & is_released & has_object_contact
        is_success_state = is_slow & is_not_spinning & has_bowl_support
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
            "Episode_Bowl/no_finger_contact_rate": torch.logical_not(has_finger_contact).float().mean(),
            "Episode_Bowl/bowl_support_rate": has_bowl_support.float().mean(),
            "Episode_Bowl/released_rate": is_released.float().mean(),
            "Episode_Bowl/raw_success_state_at_end_rate": is_success_state.float().mean(),
            "Episode_Bowl/success_5_step_rate": self._episode_success_5_hit[env_ids].mean(),
            "Episode_Bowl/success_rate": self._episode_success_10_hit[env_ids].mean(),
            "Episode_Bowl/max_success_dwell_steps": self._episode_max_success_dwell_steps[env_ids].mean(),
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
        lift_start_xy = self._masked_vector_mean(start_object_xy, lift_threshold_hit)
        no_lift_start_xy = self._masked_vector_mean(start_object_xy, failed_to_lift)
        lift_count = lift_threshold_hit.float().sum()
        no_lift_count = failed_to_lift.float().sum()
        terminal_object_z = get_object_position(self)[env_ids, 2]
        max_object_z = self._episode_max_object_z[env_ids]
        post_lift_step_count = torch.clamp(self._episode_post_lift_threshold_step_count[env_ids], min=1.0)
        lift_retention = self._episode_above_lift_threshold_step_count[env_ids] / post_lift_step_count
        first_lift_threshold_count = (self._episode_first_lift_threshold_step[env_ids] >= 0.0).float().sum()
        action_delta_count = torch.clamp(self._episode_action_delta_count[env_ids], min=1.0)
        arm_action_delta_rms = torch.sqrt(self._episode_arm_action_delta_sq_sum[env_ids] / action_delta_count)
        gripper_switch_rate = self._episode_gripper_switch_count[env_ids] / action_delta_count
        start_x_offset = start_object_xy[:, 0] - OBJECT_START_POSITION[0]
        start_y_offset = start_object_xy[:, 1] - OBJECT_START_POSITION[1]

        return {
            "Episode_Diagnostics/reset_env_count": torch.as_tensor(
                len(env_ids), device=self.device, dtype=torch.float32
            ),
            "Episode_Funnel/verified_grasp_count": self._episode_verified_grasp_hit[env_ids].sum(),
            "Episode_Funnel/lift_count": self._episode_funnel_lift_hit[env_ids].sum(),
            "Episode_Funnel/broad_entry_count": self._episode_funnel_broad_entry_hit[env_ids].sum(),
            "Episode_Funnel/centered_count": self._episode_funnel_centered_hit[env_ids].sum(),
            "Episode_Funnel/lowered_count": self._episode_funnel_lowered_hit[env_ids].sum(),
            "Episode_Funnel/opened_count": self._episode_funnel_opened_hit[env_ids].sum(),
            "Episode_Funnel/released_count": self._episode_funnel_released_hit[env_ids].sum(),
            "Episode_Funnel/supported_settled_count": self._episode_funnel_supported_settled_hit[env_ids].sum(),
            "Episode_Success/success_5_count": self._episode_success_5_hit[env_ids].sum(),
            "Episode_Success/success_10_count": self._episode_success_10_hit[env_ids].sum(),
            "Episode_Diagnostics/max_object_z": self._episode_max_object_z[env_ids].mean(),
            "Episode_Diagnostics/terminal_object_z_lift_mean": self._masked_mean(
                terminal_object_z, lift_threshold_hit
            ),
            "Episode_Diagnostics/max_object_z_lift_mean": self._masked_mean(max_object_z, lift_threshold_hit),
            "Episode_Diagnostics/lift_retention_episode_mean_lift": self._masked_mean(
                lift_retention, lift_threshold_hit
            ),
            "Episode_Diagnostics/terminal_above_lift_threshold_rate_lift": self._masked_mean(
                (terminal_object_z > OBJECT_LIFTED_HEIGHT).float(), lift_threshold_hit
            ),
            "Episode_Diagnostics/fell_below_lift_threshold_after_lift_rate_lift": self._masked_mean(
                self._episode_fell_below_lift_threshold[env_ids], lift_threshold_hit
            ),
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
            "Episode_Diagnostics/min_xy_distance_after_lift": self._masked_mean(
                torch.where(
                    lift_threshold_hit,
                    self._episode_min_xy_distance_after_lift[env_ids],
                    torch.zeros_like(self._episode_min_xy_distance_after_lift[env_ids]),
                ),
                lift_threshold_hit,
            ),
            "Episode_Diagnostics/bowl_radius_hit_after_lift_rate": (
                self._episode_bowl_radius_hit_after_lift[env_ids].mean()
            ),
            "Episode_Diagnostics/tight_radius_hit_after_lift_rate": (
                self._episode_tight_radius_hit_after_lift[env_ids].mean()
            ),
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
            "Episode_Diagnostics/verified_grasp_hit_rate": self._episode_verified_grasp_hit[env_ids].mean(),
            "Episode_Diagnostics/verified_grasp_step_fraction": (
                self._episode_verified_grasp_count[env_ids] / step_count
            ).mean(),
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
            "Episode_Diagnostics/start_object_x_mean_lift": lift_start_xy[0],
            "Episode_Diagnostics/start_object_y_mean_lift": lift_start_xy[1],
            "Episode_Diagnostics/start_object_x_mean_no_lift": no_lift_start_xy[0],
            "Episode_Diagnostics/start_object_y_mean_no_lift": no_lift_start_xy[1],
            "Episode_Diagnostics/lift_episode_count": lift_count,
            "Episode_Diagnostics/no_lift_episode_count": no_lift_count,
            "Episode_Diagnostics/no_lift_start_x_negative_offset_rate": self._masked_mean(
                (start_x_offset < 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/no_lift_start_x_positive_offset_rate": self._masked_mean(
                (start_x_offset >= 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/no_lift_start_y_negative_offset_rate": self._masked_mean(
                (start_y_offset < 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/no_lift_start_y_positive_offset_rate": self._masked_mean(
                (start_y_offset >= 0.0).float(), failed_to_lift
            ),
            "Episode_Diagnostics/arm_action_delta_rms": arm_action_delta_rms.mean(),
            "Episode_Diagnostics/arm_action_delta_rms_lift": self._masked_mean(
                arm_action_delta_rms, lift_threshold_hit
            ),
            "Episode_Diagnostics/arm_action_delta_rms_no_lift": self._masked_mean(
                arm_action_delta_rms, failed_to_lift
            ),
            "Episode_Diagnostics/gripper_switch_rate": gripper_switch_rate.mean(),
            "Episode_Diagnostics/gripper_switch_rate_lift": self._masked_mean(
                gripper_switch_rate, lift_threshold_hit
            ),
            "Episode_Diagnostics/gripper_switch_rate_no_lift": self._masked_mean(
                gripper_switch_rate, failed_to_lift
            ),
        }

    def _compute_end_reason_diagnostics(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Count mutually exclusive episode outcomes for evaluator consistency checks."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        success = self.termination_manager.get_term("object_in_bowl")[env_ids]
        drop = torch.logical_not(success) & self.termination_manager.get_term("object_dropping")[env_ids]
        timeout = (
            torch.logical_not(success)
            & torch.logical_not(drop)
            & self.termination_manager.get_term("time_out")[env_ids]
        )
        other = torch.logical_not(success | drop | timeout)
        return {
            "Episode_End/completed_count": torch.as_tensor(len(env_ids), device=self.device, dtype=torch.float32),
            "Episode_End/success_count": success.float().sum(),
            "Episode_End/drop_count": drop.float().sum(),
            "Episode_End/timeout_count": timeout.float().sum(),
            "Episode_End/other_count": other.float().sum(),
        }

    def _compute_reward_sum_diagnostics(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Expose each manager reward term's raw episode sum before reset clears it."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        episode_sums = getattr(self.reward_manager, "_episode_sums", {})
        return {
            f"Episode_Reward_Sum/{term_name}_sum": values[env_ids].sum()
            for term_name, values in episode_sums.items()
        }

    def _init_episode_diagnostic_buffers(self):
        """Allocate buffers that summarize what happened during each episode."""
        self._episode_start_object_z = torch.zeros(self.num_envs, device=self.device)
        self._episode_start_object_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_verified_grasp_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_command_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_close_near_object_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_verified_grasp_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_lift_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_broad_entry_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_centered_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_lowered_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_opened_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_released_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_funnel_supported_settled_hit = torch.zeros(self.num_envs, device=self.device)
        self._success_dwell_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_max_success_dwell_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_success_5_hit = torch.zeros(self.num_envs, device=self.device)
        self._episode_success_10_hit = torch.zeros(self.num_envs, device=self.device)
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
        self._episode_post_lift_threshold_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_above_lift_threshold_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_fell_below_lift_threshold = torch.zeros(self.num_envs, device=self.device)
        self._episode_min_xy_distance_after_lift = torch.full((self.num_envs,), torch.inf, device=self.device)
        self._episode_bowl_radius_hit_after_lift = torch.zeros(self.num_envs, device=self.device)
        self._episode_tight_radius_hit_after_lift = torch.zeros(self.num_envs, device=self.device)
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
        self._episode_verified_grasp_count[env_ids] = 0.0
        self._episode_close_command_hit[env_ids] = 0.0
        self._episode_close_near_object_hit[env_ids] = 0.0
        self._episode_verified_grasp_hit[env_ids] = 0.0
        self._episode_funnel_lift_hit[env_ids] = 0.0
        self._episode_funnel_broad_entry_hit[env_ids] = 0.0
        self._episode_funnel_centered_hit[env_ids] = 0.0
        self._episode_funnel_lowered_hit[env_ids] = 0.0
        self._episode_funnel_opened_hit[env_ids] = 0.0
        self._episode_funnel_released_hit[env_ids] = 0.0
        self._episode_funnel_supported_settled_hit[env_ids] = 0.0
        self._success_dwell_steps[env_ids] = 0.0
        self._episode_max_success_dwell_steps[env_ids] = 0.0
        self._episode_success_5_hit[env_ids] = 0.0
        self._episode_success_10_hit[env_ids] = 0.0
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
        self._episode_post_lift_threshold_step_count[env_ids] = 0.0
        self._episode_above_lift_threshold_step_count[env_ids] = 0.0
        self._episode_fell_below_lift_threshold[env_ids] = 0.0
        self._episode_min_xy_distance_after_lift[env_ids] = torch.inf
        self._episode_bowl_radius_hit_after_lift[env_ids] = 0.0
        self._episode_tight_radius_hit_after_lift[env_ids] = 0.0
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

    def _get_successful_lift_height_samples(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Return raw successful-episode heights for exact evaluation statistics."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        successful = self._episode_lift_threshold_hit[env_ids] > 0.0
        return {
            "terminal_object_z": get_object_position(self)[env_ids, 2][successful].detach().clone(),
            "max_object_z": self._episode_max_object_z[env_ids][successful].detach().clone(),
        }

    def _get_episode_samples(self, env_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """Return raw endpoint and closest-approach samples for exact distributions."""
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        object_position = get_object_position(self)[env_ids]
        target = get_placement_target_position(self, PLACEMENT_TARGET_POSITION)[env_ids]
        lifted = self._episode_lift_threshold_hit[env_ids] > 0.0
        return {
            "terminal_object_x": object_position[:, 0].detach().clone(),
            "terminal_object_y": object_position[:, 1].detach().clone(),
            "terminal_object_z": object_position[:, 2].detach().clone(),
            "terminal_xy_distance_to_bowl": torch.linalg.norm(
                object_position[:, :2] - target[:, :2], dim=1
            )
            .detach()
            .clone(),
            "max_object_z": self._episode_max_object_z[env_ids].detach().clone(),
            "min_xy_distance_after_lift": self._episode_min_xy_distance_after_lift[env_ids][lifted]
            .detach()
            .clone(),
        }

    def _masked_vector_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Average vector values only over true mask entries, returning zeros if empty."""
        mask = mask.float().unsqueeze(-1)
        count = mask.sum()
        return torch.where(
            count > 0.0,
            (values * mask).sum(dim=0) / torch.clamp(count, min=1.0),
            torch.zeros(values.shape[1], device=self.device),
        )
