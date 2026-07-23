# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import combine_frame_transforms

from .observations import get_ee_position, get_object_position, get_placement_target_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Updates per-episode diagnostic buffers without changing the reward.
def update_episode_diagnostics(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Update diagnostic buffers and return zero reward."""
    if hasattr(env, "update_episode_diagnostics"):
        env.update_episode_diagnostics()
    return torch.zeros(env.num_envs, device=env.device)


# Checks if the cube has been lifted off the table.
def check_object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the object has cleared the table by enough height."""
    object_position = get_object_position(env, object_cfg)
    return object_position[:, 2] > minimal_height


# Checks if the lifted cube is over the target area.
def check_object_above_target(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the lifted object is over the placement target's XY radius."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    return torch.logical_and(xy_distance < radius, object_position[:, 2] > minimal_height)


# Checks if the cube is inside the bowl and mostly settled.
def check_object_in_bowl(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    min_height: float,
    max_height: float,
    max_speed: float,
    max_angular_speed: float,
    min_gripper_open: float,
    support_force_threshold: float,
    finger_contact_force_threshold: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the released object is settled and supported inside the bowl."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_speed = torch.linalg.norm(object_asset.data.root_lin_vel_w[:, :3], dim=1)
    object_angular_speed = torch.linalg.norm(object_asset.data.root_ang_vel_w[:, :3], dim=1)
    finger_joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids]
    is_settled = object_speed < max_speed
    is_not_spinning = object_angular_speed < max_angular_speed
    is_gripper_open = torch.all(finger_joint_pos > min_gripper_open, dim=1)
    is_released = is_gripper_open & torch.logical_not(
        check_finger_object_contact(env, finger_contact_force_threshold)
    )
    is_bowl_supported = is_released & check_bowl_support_contact(
        env,
        target_position,
        radius,
        min_height,
        max_height,
        support_force_threshold,
        object_cfg=object_cfg,
    )
    return is_settled & is_not_spinning & is_bowl_supported


# Rewards the hand for getting close to the cube before it has been lifted.
def compute_reaching_object_reward(
    env: ManagerBasedRLEnv,
    std: float,
    disable_after_lift_height: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the hand for getting near the object, optionally only before lift."""
    ee_position = get_ee_position(env)
    object_position = get_object_position(env, object_cfg)
    distance = torch.linalg.norm(ee_position - object_position, dim=1)
    reward = 1.0 - torch.tanh(distance / std)
    if disable_after_lift_height is None:
        return reward
    not_lifted = torch.logical_not(check_object_lifted(env, disable_after_lift_height, object_cfg))
    return reward * not_lifted.float()


# Rewards closing the gripper only when the hand is already near the cube.
def compute_grasping_object_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward near-object gripper closing before the object has been lifted."""
    ee_position = get_ee_position(env)
    object_position = get_object_position(env, object_cfg)
    distance = torch.linalg.norm(ee_position - object_position, dim=1)
    near_object = 1.0 - torch.tanh(distance / std)
    gripper_action = env.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
    closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0)
    not_lifted = torch.logical_not(check_object_lifted(env, minimal_height, object_cfg))
    return near_object * closing_gripper * not_lifted.float()


def check_verified_grasp(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    history_length: int,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_object_contact"),
) -> torch.Tensor:
    """Check for sustained bilateral finger contact while the gripper is closing."""

    def has_sustained_contact(sensor_cfg: SceneEntityCfg) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        force_history = sensor.data.force_matrix_w_history[:, :history_length]
        force_magnitude = torch.linalg.vector_norm(force_history, dim=-1)
        return torch.all(force_magnitude > force_threshold, dim=(1, 2, 3))

    gripper_action = env.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
    is_closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0) > 0.0
    return has_sustained_contact(left_sensor_cfg) & has_sustained_contact(right_sensor_cfg) & is_closing_gripper


def check_filtered_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Check whether a filtered contact sensor currently exceeds a force threshold."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if sensor.data.force_matrix_w is None:
        raise RuntimeError(f"Contact sensor '{sensor_cfg.name}' has no filtered force data.")
    force_magnitude = torch.linalg.vector_norm(sensor.data.force_matrix_w, dim=-1)
    contact_dims = tuple(range(1, force_magnitude.ndim))
    return torch.any(force_magnitude > force_threshold, dim=contact_dims)


def check_finger_object_contact(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("left_finger_object_contact"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("right_finger_object_contact"),
) -> torch.Tensor:
    """Check whether either gripper finger still contacts the object."""
    return check_filtered_contact(env, force_threshold, left_sensor_cfg) | check_filtered_contact(
        env, force_threshold, right_sensor_cfg
    )


def check_bowl_support_contact(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    min_height: float,
    max_height: float,
    force_threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("object_bowl_support_contact"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the object has support contact inside the bowl region."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    is_inside_radius = xy_distance < radius
    is_inside_height = torch.logical_and(object_position[:, 2] > min_height, object_position[:, 2] < max_height)
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    force_magnitude = torch.linalg.vector_norm(sensor.data.net_forces_w, dim=-1)
    has_contact = torch.any(force_magnitude > force_threshold, dim=1)
    return is_inside_radius & is_inside_height & has_contact


# Rewards verified bilateral contact before the cube reaches full lift.
def compute_verified_grasp_reward(
    env: ManagerBasedRLEnv,
    force_threshold: float,
    history_length: int,
    initial_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a sustained bilateral grasp, tapering to zero at full lift."""
    is_verified_grasp = check_verified_grasp(env, force_threshold, history_length)
    lift_progress = compute_object_height_progress_reward(env, initial_height, target_height, object_cfg)
    return is_verified_grasp.float() * (1.0 - lift_progress)


# Rewards smooth progress as the cube rises from the table.
def compute_object_height_progress_reward(
    env: ManagerBasedRLEnv,
    initial_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object height as a bounded 0-to-1 lift progress signal."""
    object_position = get_object_position(env, object_cfg)
    lift_range = target_height - initial_height
    return torch.clamp((object_position[:, 2] - initial_height) / lift_range, min=0.0, max=1.0)


# Rewards lift progress only during plausible grasp attempts.
def compute_gated_object_height_progress_reward(
    env: ManagerBasedRLEnv,
    initial_height: float,
    target_height: float,
    near_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lift progress when the hand is near the object and the gripper is closing."""
    ee_position = get_ee_position(env)
    object_position = get_object_position(env, object_cfg)
    distance = torch.linalg.norm(ee_position - object_position, dim=1)
    gripper_action = env.action_manager.get_term("gripper_action").raw_actions.squeeze(-1)
    is_near_object = distance < near_distance
    is_closing_gripper = torch.clamp(-gripper_action, min=0.0, max=1.0) > 0.0
    lift_range = target_height - initial_height
    lift_progress = torch.clamp((object_position[:, 2] - initial_height) / lift_range, min=0.0, max=1.0)
    return lift_progress * (is_near_object & is_closing_gripper).float()


# Rewards the cube for clearing the table.
def compute_object_lifted_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Give a capped reward once the object is lifted; higher is not better."""
    return check_object_lifted(env, minimal_height, object_cfg).float()


# Rewards the cube for moving toward a 3D target position.
def compute_object_to_target_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward full 3D object progress toward the target after it clears a low height gate."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    distance = torch.linalg.norm(object_position - target, dim=1)
    is_high_enough = check_object_lifted(env, minimal_height, object_cfg)
    return (1.0 - torch.tanh(distance / std)) * is_high_enough.float()


# Rewards the cube for moving toward the sampled lift command.
def compute_object_goal_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    gate_target_position: tuple[float, float, float] | None = None,
    gate_radius: float = 0.0,
    gate_minimal_height: float = 0.0,
    gate_reward_scale: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object distance to the command target, matching the official lift task."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    desired_pos_b = command[:, :3]
    desired_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, desired_pos_b)
    distance = torch.linalg.norm(desired_pos_w - object_asset.data.root_pos_w[:, :3], dim=1)
    is_high_enough = object_asset.data.root_pos_w[:, 2] > minimal_height
    reward = (1.0 - torch.tanh(distance / std)) * is_high_enough.float()
    if gate_target_position is None:
        return reward
    is_over_gate_target = check_object_above_target(
        env, gate_target_position, gate_radius, gate_minimal_height, object_cfg
    )
    return torch.where(is_over_gate_target, reward * gate_reward_scale, reward)


# Rewards the lifted cube for moving toward the target in XY.
def compute_object_to_target_xy_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    std: float,
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward XY progress toward the target after the object is lifted."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    is_lifted = check_object_lifted(env, minimal_height, object_cfg)
    is_above_target = check_object_above_target(env, target_position, radius, minimal_height, object_cfg)
    return (1.0 - torch.tanh(xy_distance / std)) * is_lifted.float() * torch.logical_not(is_above_target).float()


# Rewards the lifted cube for moving toward the target in XY, keeping max reward inside the target radius.
def compute_saturated_object_to_target_xy_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    std: float,
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward post-lift XY transport and saturate once the object reaches the target area."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    base_reward = 1.0 - torch.tanh(xy_distance / std)
    radius_reward = 1.0 - torch.tanh(torch.as_tensor(radius, device=env.device) / std)
    saturated_reward = torch.clamp(base_reward / radius_reward, max=1.0)
    is_lifted = check_object_lifted(env, minimal_height, object_cfg)
    return saturated_reward * is_lifted.float()


class ComputeObjectToTargetXYProgressReward(ManagerTermBase):
    """Reward movement toward the target instead of paying for standing near it."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(self.num_envs, device=self.device)
        self._was_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = 0.0
        self._was_lifted[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_position: tuple[float, float, float],
        std: float,
        radius: float,
        minimal_height: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
        """Return positive reward for approaching, zero for holding, and negative reward for retreating."""
        object_position = get_object_position(env, object_cfg)
        target = get_placement_target_position(env, target_position)
        xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
        base_potential = 1.0 - torch.tanh(xy_distance / std)
        radius_potential = 1.0 - torch.tanh(torch.as_tensor(radius, device=env.device) / std)
        potential = torch.clamp(base_potential / radius_potential, max=1.0)
        is_lifted = check_object_lifted(env, minimal_height, object_cfg)

        reward = (potential - self._previous_potential) / env.step_dt
        reward = torch.where(is_lifted & self._was_lifted, reward, 0.0)

        self._previous_potential.copy_(potential)
        self._was_lifted.copy_(is_lifted)
        return reward


# Rewards the current success milestone.
def compute_object_above_target_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the current milestone: lifted object reaches the target XY area."""
    return check_object_above_target(env, target_position, radius, minimal_height, object_cfg).float()


# Rewards the cube for lowering toward bowl height after it is over the bowl.
def compute_object_lowering_into_bowl_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    target_height: float,
    height_std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the bridge from high carry to low placement inside the bowl."""
    object_position = get_object_position(env, object_cfg)
    target = get_placement_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    height_distance = torch.abs(object_position[:, 2] - target_height)
    is_over_target = xy_distance < radius
    is_lifted = check_object_lifted(env, minimal_height, object_cfg)
    return (1.0 - torch.tanh(height_distance / height_std)) * is_over_target.float() * is_lifted.float()


# Rewards the single timestep when a named success termination fires.
def compute_termination_reward(env: ManagerBasedRLEnv, termination_name: str) -> torch.Tensor:
    """Return one for environments whose named termination fired this step."""
    return env.termination_manager.get_term(termination_name).float()


# Rewards the cube for ending up inside the bowl.
def compute_object_in_bowl_success_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    min_height: float,
    max_height: float,
    max_speed: float,
    max_angular_speed: float,
    min_gripper_open: float,
    support_force_threshold: float,
    finger_contact_force_threshold: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the successful placement state."""
    return check_object_in_bowl(
        env,
        target_position,
        radius,
        min_height,
        max_height,
        max_speed,
        max_angular_speed,
        min_gripper_open,
        support_force_threshold,
        finger_contact_force_threshold,
        robot_cfg,
        object_cfg,
    ).float()


# Stops the episode once the cube is successfully in the bowl.
def terminate_on_object_in_bowl_success(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    min_height: float,
    max_height: float,
    max_speed: float,
    max_angular_speed: float,
    min_gripper_open: float,
    support_force_threshold: float,
    finger_contact_force_threshold: float,
    dwell_steps: int,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """End the episode when the object is inside the bowl and settled."""
    is_success_state = check_object_in_bowl(
        env,
        target_position,
        radius,
        min_height,
        max_height,
        max_speed,
        max_angular_speed,
        min_gripper_open,
        support_force_threshold,
        finger_contact_force_threshold,
        robot_cfg,
        object_cfg,
    )
    if hasattr(env, "update_success_dwell"):
        return env.update_success_dwell(is_success_state, dwell_steps)
    return is_success_state
