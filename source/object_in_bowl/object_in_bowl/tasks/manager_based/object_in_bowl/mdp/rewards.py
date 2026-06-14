# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import get_ee_position, get_object_position, get_placement_target_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


# Rewards the hand for getting close to the cube before lifting it.
def compute_reaching_object_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the hand for getting near the object before the object is lifted."""
    ee_position = get_ee_position(env)
    object_position = get_object_position(env, object_cfg)
    distance = torch.linalg.norm(ee_position - object_position, dim=1)
    not_lifted = torch.logical_not(check_object_lifted(env, minimal_height, object_cfg))
    return (1.0 - torch.tanh(distance / std)) * not_lifted.float()


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


# Rewards the cube for clearing the table.
def compute_object_lifted_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Give a capped reward once the object is lifted; higher is not better."""
    return check_object_lifted(env, minimal_height, object_cfg).float()


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
