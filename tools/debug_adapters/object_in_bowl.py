"""Object-in-bowl diagnostics for the generic Isaac Lab debug viewer."""

from __future__ import annotations

from typing import Any

from .base import (
    distance,
    fmt_number,
    number,
    record_error,
    stage,
    to_jsonable,
    to_scalar,
    xy_distance,
)


ADAPTER_NAME = "object_in_bowl"
DISPLAY_NAME = "Object-in-bowl adapter"
DESCRIPTION = "Shows task-specific stages for reaching, lifting, carrying, settling, and release."
NEAR_OBJECT_DISTANCE = 0.08


def matches(task: str | None) -> bool:
    """Return true when this adapter should handle the current task name."""
    normalized = (task or "").lower().replace("-", "_")
    return "object_in_bowl" in normalized


def collect(env: Any, snapshot: dict[str, Any], env_index: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    """Collect object-in-bowl-specific diagnostics for the generic viewer slot."""
    signals = _collect_signals(env, snapshot, env_index, errors)
    return {
        "adapterName": ADAPTER_NAME,
        "displayName": DISPLAY_NAME,
        "description": DESCRIPTION,
        "stages": _collect_stages(snapshot, signals),
        "signals": signals,
        "episodeSignals": _collect_episode_signals(env, env_index),
        "rewardProbes": _collect_reward_probes(env, env_index, errors),
    }


def build_mock_diagnostics(
    cube: list[float],
    ee: list[float],
    target: list[float],
    mock_gripper_command: float,
    viewer_mode: str,
) -> dict[str, Any]:
    """Build fake object-in-bowl diagnostics for browser-only screenshots."""
    object_target_distance = distance(cube, target)
    mock_lift_progress = max(0.0, min(1.0, (cube[2] - 0.055) / 0.05))
    mock_lift_gate_active = viewer_mode == "policy"
    return {
        "adapterName": ADAPTER_NAME,
        "displayName": DISPLAY_NAME,
        "description": DESCRIPTION,
        "stages": [
            stage("near_object", "Hand near cube", True, "ee-object 0.045m, target < 0.080m", 0.045, 0.035, "m"),
            stage(
                "close_command",
                "Close command",
                viewer_mode == "policy",
                f"gripper command {mock_gripper_command:.3f}",
                mock_gripper_command,
            ),
            stage(
                "close_near_object",
                "Close while near cube",
                viewer_mode == "policy",
                "close command and hand-near-cube are both true",
            ),
            stage(
                "object_lifted",
                "Cube lifted",
                cube[2] > 0.105,
                "cube z is above lift threshold",
                cube[2],
                cube[2] - 0.105,
                "m",
            ),
            stage(
                "over_bowl",
                "Cube over bowl",
                False,
                "xy distance still outside bowl radius",
                object_target_distance,
                0.11 - (object_target_distance or 0.0),
                "m",
            ),
            stage(
                "inside_bowl_height",
                "Cube at bowl height",
                False,
                "cube is not yet at bowl placement height",
                cube[2],
                min(cube[2] - 0.044, 0.109 - cube[2]),
                "m",
            ),
            stage("settled", "Cube settled", True, "mock cube speed is low", None, 0.26),
            stage("released", "Gripper released", False, "mock gripper is not released", 0.012, -0.018, "m"),
            stage("success_gate", "Success gate", False, "over bowl, correct height, settled, and released"),
        ],
        "signals": {
            "objectZ": cube[2],
            "eeObjectDistance": distance(ee, cube),
            "objectTargetDistance": object_target_distance,
            "gripperCommand": mock_gripper_command,
            "gripperOpening": 0.025,
            "objectLinearSpeed": 0.04,
        },
        "episodeSignals": {
            "max_lift_progress": max(0.0, min(1.0, (cube[2] - 0.055) / 0.05)),
            "min_ee_object_distance": 0.045,
            "close_near_object_hit": 1.0 if viewer_mode == "policy" else 0.0,
        },
        "rewardProbes": [
            {
                "name": "grasping_object",
                "label": "Would grasp reward fire?",
                "probeKind": "inactive_candidate",
                "statusLabel": "candidate",
                "isActiveReward": False,
                "currentValue": 0.5 if viewer_mode == "policy" else 0.0,
                "wouldFire": viewer_mode == "policy",
                "detail": "near cube + closing gripper before lift",
                "source": "object_in_bowl_adapter",
            },
            {
                "name": "object_lift_progress",
                "label": "Would gated lift-progress reward fire?",
                "probeKind": "inactive_candidate",
                "statusLabel": "candidate",
                "isActiveReward": False,
                "currentValue": mock_lift_progress if mock_lift_gate_active else 0.0,
                "wouldFire": mock_lift_gate_active and cube[2] > 0.055,
                "detail": "lift progress gated by near cube + closing gripper",
                "source": "object_in_bowl_adapter",
            },
        ],
    }


def _find_asset(snapshot: dict[str, Any], name: str) -> dict[str, Any] | None:
    for asset in snapshot.get("scene", {}).get("assets", []):
        if asset.get("name") == name:
            return asset
    return None


def _reward_param(snapshot: dict[str, Any], reward_name: str, param_name: str) -> Any:
    for reward in snapshot.get("rewards", []):
        if reward.get("name") == reward_name:
            return (reward.get("params") or {}).get(param_name)
    return None


def _termination_param(snapshot: dict[str, Any], termination_name: str, param_name: str) -> Any:
    for termination in snapshot.get("terminations", []):
        if termination.get("name") == termination_name:
            return (termination.get("params") or {}).get(param_name)
    return None


def _get_tensor_scalar(value: Any, env_index: int) -> float | None:
    return number(to_jsonable(value, env_index))


def _collect_gripper_signals(env: Any, env_index: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    try:
        robot = env.scene["robot"]
        finger_cfg = getattr(env, "_finger_robot_cfg", None)
        joint_ids = getattr(finger_cfg, "joint_ids", None)
        if joint_ids is not None:
            finger_joint_pos = robot.data.joint_pos[env_index, joint_ids]
            signals["fingerPositions"] = to_jsonable(finger_joint_pos, None, max_items=4)
            signals["gripperOpening"] = _get_tensor_scalar(finger_joint_pos.sum(), 0)
            signals["minFingerPosition"] = _get_tensor_scalar(finger_joint_pos.min(), 0)
    except Exception as exc:
        record_error(errors, "object_in_bowl.gripper_joint_signals", exc)

    try:
        raw_actions = env.action_manager.get_term("gripper_action").raw_actions
        action = to_jsonable(raw_actions, env_index, max_items=1)
        if isinstance(action, list) and action:
            action = action[0]
        signals["gripperCommand"] = action
        signals["closeCommandActive"] = (number(action) or 0.0) < 0.0
    except Exception as exc:
        record_error(errors, "object_in_bowl.gripper_action_signals", exc)
    return signals


def _collect_object_motion(env: Any, env_index: int, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        object_asset = env.scene["object"]
        linear_speed = object_asset.data.root_lin_vel_w[env_index, :3].norm()
        angular_speed = object_asset.data.root_ang_vel_w[env_index, :3].norm()
        return {
            "objectLinearSpeed": _get_tensor_scalar(linear_speed, 0),
            "objectAngularSpeed": _get_tensor_scalar(angular_speed, 0),
        }
    except Exception as exc:
        record_error(errors, "object_in_bowl.object_motion_signals", exc)
        return {}


def _collect_signals(
    env: Any, snapshot: dict[str, Any], env_index: int, errors: list[dict[str, str]]
) -> dict[str, Any]:
    signals = dict(snapshot.get("metrics", {}))
    signals.update(_collect_gripper_signals(env, env_index, errors))
    signals.update(_collect_object_motion(env, env_index, errors))
    return signals


def _collect_episode_signals(env: Any, env_index: int) -> dict[str, Any]:
    signal_names = (
        "_episode_max_lift_progress",
        "_episode_max_object_z_delta",
        "_episode_min_ee_object_distance",
        "_episode_min_gripper_opening",
        "_episode_close_command_hit",
        "_episode_close_near_object_hit",
        "_episode_max_lift_progress_after_close_near_object",
        "_episode_max_lift_progress_while_gate_active",
        "_episode_gate_active_at_max_lift_progress",
        "_episode_near_object_at_max_lift_progress",
        "_episode_close_command_at_max_lift_progress",
        "_episode_first_close_near_step",
        "_episode_first_lift_threshold_step",
        "_episode_lift_threshold_hit",
    )
    signals = {}
    for name in signal_names:
        value = getattr(env, name, None)
        if value is not None:
            signals[name.removeprefix("_episode_")] = to_jsonable(value, env_index)
    return signals


def _collect_stages(snapshot: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    object_asset = _find_asset(snapshot, "object")
    object_position = object_asset.get("position") if object_asset else None
    target_position = _termination_param(snapshot, "object_in_bowl", "target_position")
    lifted_height = _reward_param(snapshot, "lifting_object", "minimal_height")
    bowl_radius = _termination_param(snapshot, "object_in_bowl", "radius")
    bowl_min_height = _termination_param(snapshot, "object_in_bowl", "min_height")
    bowl_max_height = _termination_param(snapshot, "object_in_bowl", "max_height")
    max_speed = _termination_param(snapshot, "object_in_bowl", "max_speed")
    max_angular_speed = _termination_param(snapshot, "object_in_bowl", "max_angular_speed")
    min_gripper_open = _termination_param(snapshot, "object_in_bowl", "min_gripper_open")

    ee_distance = number(metrics.get("eeObjectDistance"))
    object_z = number(metrics.get("objectZ"))
    xy_dist = xy_distance(object_position, target_position)
    linear_speed = number(metrics.get("objectLinearSpeed"))
    angular_speed = number(metrics.get("objectAngularSpeed"))
    min_finger_position = number(metrics.get("minFingerPosition"))
    close_command_active = metrics.get("closeCommandActive")

    near_object = ee_distance is not None and ee_distance < NEAR_OBJECT_DISTANCE
    lifted = object_z is not None and number(lifted_height) is not None and object_z > float(lifted_height)
    over_bowl = xy_dist is not None and number(bowl_radius) is not None and xy_dist < float(bowl_radius)
    inside_height = (
        object_z is not None
        and number(bowl_min_height) is not None
        and number(bowl_max_height) is not None
        and float(bowl_min_height) < object_z < float(bowl_max_height)
    )
    settled = (
        linear_speed is not None
        and angular_speed is not None
        and number(max_speed) is not None
        and number(max_angular_speed) is not None
        and linear_speed < float(max_speed)
        and angular_speed < float(max_angular_speed)
    )
    gripper_open_for_release = (
        min_finger_position is not None
        and number(min_gripper_open) is not None
        and min_finger_position > float(min_gripper_open)
    )

    stages = [
        stage(
            "near_object",
            "Hand near cube",
            near_object if ee_distance is not None else None,
            f"ee-object {fmt_number(ee_distance)}m, target < {NEAR_OBJECT_DISTANCE:.3f}m",
            ee_distance,
            NEAR_OBJECT_DISTANCE - ee_distance if ee_distance is not None else None,
            "m",
        ),
        stage(
            "close_command",
            "Close command",
            bool(close_command_active) if close_command_active is not None else None,
            f"gripper command {fmt_number(metrics.get('gripperCommand'))}",
            metrics.get("gripperCommand"),
        ),
        stage(
            "close_near_object",
            "Close while near cube",
            (
                (bool(close_command_active) and near_object)
                if close_command_active is not None and ee_distance is not None
                else None
            ),
            "close command and hand-near-cube are both true",
        ),
        stage(
            "object_lifted",
            "Cube lifted",
            lifted if object_z is not None and lifted_height is not None else None,
            f"cube z {fmt_number(object_z)}m, lift threshold {fmt_number(lifted_height)}m",
            object_z,
            object_z - float(lifted_height)
            if object_z is not None and number(lifted_height) is not None
            else None,
            "m",
        ),
        stage(
            "over_bowl",
            "Cube over bowl",
            over_bowl if xy_dist is not None and bowl_radius is not None else None,
            f"xy distance {fmt_number(xy_dist)}m, radius {fmt_number(bowl_radius)}m",
            xy_dist,
            float(bowl_radius) - xy_dist if xy_dist is not None and number(bowl_radius) is not None else None,
            "m",
        ),
        stage(
            "inside_bowl_height",
            "Cube at bowl height",
            inside_height if object_z is not None and bowl_min_height is not None and bowl_max_height is not None else None,
            f"cube z {fmt_number(object_z)}m, allowed {fmt_number(bowl_min_height)}-{fmt_number(bowl_max_height)}m",
            object_z,
            _height_margin(object_z, bowl_min_height, bowl_max_height),
            "m",
        ),
        stage(
            "settled",
            "Cube settled",
            settled if linear_speed is not None and angular_speed is not None else None,
            f"speed {fmt_number(linear_speed)}m/s, angular {fmt_number(angular_speed)}rad/s",
            None,
            _settled_margin(linear_speed, angular_speed, max_speed, max_angular_speed),
        ),
        stage(
            "released",
            "Gripper released",
            gripper_open_for_release if min_finger_position is not None and min_gripper_open is not None else None,
            f"min finger {fmt_number(min_finger_position)}m, release threshold > {fmt_number(min_gripper_open)}m",
            min_finger_position,
            min_finger_position - float(min_gripper_open)
            if min_finger_position is not None and number(min_gripper_open) is not None
            else None,
            "m",
        ),
    ]
    success = all(item["active"] is True for item in stages[4:8])
    stages.append(stage("success_gate", "Success gate", success, "over bowl, correct height, settled, and released"))
    return stages


def _height_margin(object_z: float | None, min_height: Any, max_height: Any) -> float | None:
    if object_z is None or number(min_height) is None or number(max_height) is None:
        return None
    return min(object_z - float(min_height), float(max_height) - object_z)


def _settled_margin(linear_speed: float | None, angular_speed: float | None, max_speed: Any, max_angular_speed: Any):
    if (
        linear_speed is None
        or angular_speed is None
        or number(max_speed) is None
        or number(max_angular_speed) is None
    ):
        return None
    return min(float(max_speed) - linear_speed, float(max_angular_speed) - angular_speed)


def _probe_row(name: str, label: str, value: Any, detail: str, env_index: int) -> dict[str, Any]:
    scalar = to_scalar(value, env_index)
    return {
        "name": name,
        "label": label,
        "probeKind": "inactive_candidate",
        "statusLabel": "candidate",
        "isActiveReward": False,
        "currentValue": scalar,
        "wouldFire": (number(scalar) or 0.0) > 1.0e-6,
        "detail": detail,
        "source": "object_in_bowl_adapter",
    }


def _collect_reward_probes(env: Any, env_index: int, errors: list[dict[str, str]]) -> list[dict[str, Any]]:
    try:
        from isaaclab.managers import SceneEntityCfg
        from object_in_bowl import rewards
        from object_in_bowl.env_cfg import (
            BOWL_LOWERING_RADIUS,
            BOWL_LOWERING_REWARD_MIN_HEIGHT,
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
        )
    except Exception as exc:
        record_error(errors, "object_in_bowl.reward_probe_imports", exc)
        return []

    robot_cfg = SceneEntityCfg("robot", joint_names=["panda_finger.*"])
    try:
        robot_cfg.resolve(env.scene)
    except Exception as exc:
        record_error(errors, "object_in_bowl.reward_probe_robot_cfg", exc)

    probe_specs = (
        (
            "grasping_object",
            "Would grasp reward fire?",
            lambda: rewards.compute_grasping_object_reward(env, std=0.10, minimal_height=OBJECT_LIFTED_HEIGHT),
            "near cube + closing gripper before lift",
        ),
        (
            "object_lift_progress",
            "Would gated lift-progress reward fire?",
            lambda: rewards.compute_gated_object_height_progress_reward(
                env,
                initial_height=OBJECT_START_POSITION[2],
                target_height=OBJECT_LIFTED_HEIGHT,
                near_distance=LIFT_PROGRESS_NEAR_OBJECT_DISTANCE,
            ),
            "lift progress gated by near cube + closing gripper",
        ),
        (
            "object_to_bowl",
            "Would carry-to-bowl reward fire?",
            lambda: rewards.compute_object_to_target_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                std=0.30,
                minimal_height=BOWL_LOWERING_REWARD_MIN_HEIGHT,
            ),
            "cube closer to bowl target after a low lift gate",
        ),
        (
            "lowering_into_bowl",
            "Would lowering reward fire?",
            lambda: rewards.compute_object_lowering_into_bowl_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                radius=BOWL_LOWERING_RADIUS,
                target_height=BOWL_LOWERING_TARGET_HEIGHT,
                height_std=0.05,
                minimal_height=BOWL_LOWERING_REWARD_MIN_HEIGHT,
            ),
            "cube over bowl radius and near bowl height",
        ),
        (
            "object_in_bowl_success",
            "Would bowl success reward fire?",
            lambda: rewards.compute_object_in_bowl_success_reward(
                env,
                target_position=PLACEMENT_TARGET_POSITION,
                radius=BOWL_SUCCESS_RADIUS,
                min_height=BOWL_SUCCESS_MIN_HEIGHT,
                max_height=BOWL_SUCCESS_MAX_HEIGHT,
                max_speed=BOWL_SUCCESS_MAX_SPEED,
                max_angular_speed=BOWL_SUCCESS_MAX_ANGULAR_SPEED,
                min_gripper_open=BOWL_SUCCESS_MIN_GRIPPER_OPEN,
                robot_cfg=robot_cfg,
            ),
            "inside bowl radius, correct height, slow, not spinning, released",
        ),
    )
    probes = []
    for name, label, func, detail in probe_specs:
        try:
            probes.append(_probe_row(name, label, func(), detail, env_index))
        except Exception as exc:
            record_error(errors, f"object_in_bowl.reward_probe.{name}", exc)
            probes.append(
                {
                    "name": name,
                    "label": label,
                    "probeKind": "inactive_candidate",
                    "statusLabel": "candidate",
                    "isActiveReward": False,
                    "currentValue": None,
                    "wouldFire": None,
                    "detail": f"{detail}; probe unavailable: {exc}",
                    "source": "object_in_bowl_adapter",
                }
            )
    return probes
