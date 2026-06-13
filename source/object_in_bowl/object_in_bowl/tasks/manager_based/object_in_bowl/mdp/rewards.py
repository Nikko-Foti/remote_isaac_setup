# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .observations import get_ee_position

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_object_position(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Object position in each environment's local frame."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    return object_asset.data.root_pos_w[:, :3] - env.scene.env_origins


def get_target_position(env: ManagerBasedRLEnv, target_position: tuple[float, float, float]) -> torch.Tensor:
    """Placement target position repeated once per environment."""
    target = env.scene.env_origins.new_tensor(target_position)
    return target.repeat(env.scene.env_origins.shape[0], 1)


def check_object_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the object has cleared the table by enough height."""
    object_position = get_object_position(env, object_cfg)
    return object_position[:, 2] > minimal_height


def check_object_at_target(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check whether the lifted object is within the placement target's XY radius."""
    object_position = get_object_position(env, object_cfg)
    target = get_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    return torch.logical_and(xy_distance < radius, object_position[:, 2] > minimal_height)


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


def compute_object_lifted_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Give a capped reward once the object is lifted; higher is not better."""
    return check_object_lifted(env, minimal_height, object_cfg).float()


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
    target = get_target_position(env, target_position)
    xy_distance = torch.linalg.norm(object_position[:, :2] - target[:, :2], dim=1)
    is_lifted = check_object_lifted(env, minimal_height, object_cfg)
    is_at_target = check_object_at_target(env, target_position, radius, minimal_height, object_cfg)
    return (1.0 - torch.tanh(xy_distance / std)) * is_lifted.float() * torch.logical_not(is_at_target).float()


def compute_object_at_target_reward(
    env: ManagerBasedRLEnv,
    target_position: tuple[float, float, float],
    radius: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward success when the lifted object reaches the placement target."""
    return check_object_at_target(env, target_position, radius, minimal_height, object_cfg).float()
