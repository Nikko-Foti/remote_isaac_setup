# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a fixed-size clean evaluation for an RSL-RL checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint for a fixed number of episodes.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of parallel environments to simulate.")
parser.add_argument("--num_episodes", type=int, default=4096, help="Minimum completed episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=None, help="Safety cap on environment steps.")
parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import object_in_bowl.tasks  # noqa: F401


COUNT_SUFFIX = "_count"
RATIO_SPECS = {
    "Episode_Diagnostics/gate_rate_after_lift_005m": (
        "Episode_Diagnostics/post_lift_005m_gate_count",
        "Episode_Diagnostics/post_lift_005m_step_count",
    ),
    "Episode_Diagnostics/near_rate_after_lift_005m": (
        "Episode_Diagnostics/post_lift_005m_near_count",
        "Episode_Diagnostics/post_lift_005m_step_count",
    ),
    "Episode_Diagnostics/close_rate_after_lift_005m": (
        "Episode_Diagnostics/post_lift_005m_close_count",
        "Episode_Diagnostics/post_lift_005m_step_count",
    ),
    "Episode_Diagnostics/gate_off_after_lift_005m_far_component_rate": (
        "Episode_Diagnostics/post_lift_005m_gate_off_far_count",
        "Episode_Diagnostics/post_lift_005m_step_count",
    ),
    "Episode_Diagnostics/gate_off_after_lift_005m_open_component_rate": (
        "Episode_Diagnostics/post_lift_005m_gate_off_open_count",
        "Episode_Diagnostics/post_lift_005m_step_count",
    ),
    "Episode_Diagnostics/gate_rate_after_lift_020m": (
        "Episode_Diagnostics/post_lift_020m_gate_count",
        "Episode_Diagnostics/post_lift_020m_step_count",
    ),
    "Episode_Diagnostics/near_rate_after_lift_020m": (
        "Episode_Diagnostics/post_lift_020m_near_count",
        "Episode_Diagnostics/post_lift_020m_step_count",
    ),
    "Episode_Diagnostics/close_rate_after_lift_020m": (
        "Episode_Diagnostics/post_lift_020m_close_count",
        "Episode_Diagnostics/post_lift_020m_step_count",
    ),
    "Episode_Diagnostics/gate_off_after_lift_020m_far_component_rate": (
        "Episode_Diagnostics/post_lift_020m_gate_off_far_count",
        "Episode_Diagnostics/post_lift_020m_step_count",
    ),
    "Episode_Diagnostics/gate_off_after_lift_020m_open_component_rate": (
        "Episode_Diagnostics/post_lift_020m_gate_off_open_count",
        "Episode_Diagnostics/post_lift_020m_step_count",
    ),
}
WEIGHT_KEY_BY_METRIC = {
    "Episode_Diagnostics/gate_active_at_max_lift_rate_failed_partial_005m": (
        "Episode_Diagnostics/failed_partial_lift_005m_episode_count"
    ),
    "Episode_Diagnostics/near_object_at_max_lift_rate_failed_partial_005m": (
        "Episode_Diagnostics/failed_partial_lift_005m_episode_count"
    ),
    "Episode_Diagnostics/close_command_at_max_lift_rate_failed_partial_005m": (
        "Episode_Diagnostics/failed_partial_lift_005m_episode_count"
    ),
    "Episode_Diagnostics/gate_active_at_max_lift_rate_failed_partial_020m": (
        "Episode_Diagnostics/failed_partial_lift_020m_episode_count"
    ),
    "Episode_Diagnostics/near_object_at_max_lift_rate_failed_partial_020m": (
        "Episode_Diagnostics/failed_partial_lift_020m_episode_count"
    ),
    "Episode_Diagnostics/close_command_at_max_lift_rate_failed_partial_020m": (
        "Episode_Diagnostics/failed_partial_lift_020m_episode_count"
    ),
    "Episode_Diagnostics/gate_active_at_first_lift_threshold_crossing_rate": (
        "Episode_Diagnostics/first_lift_threshold_crossing_episode_count"
    ),
    "Episode_Diagnostics/close_near_before_lift_threshold_rate": (
        "Episode_Diagnostics/first_lift_threshold_crossing_episode_count"
    ),
    "Episode_Diagnostics/start_object_x_mean_success": "Episode_Diagnostics/success_episode_count",
    "Episode_Diagnostics/start_object_y_mean_success": "Episode_Diagnostics/success_episode_count",
    "Episode_Diagnostics/arm_action_delta_rms_success": "Episode_Diagnostics/success_episode_count",
    "Episode_Diagnostics/gripper_switch_rate_success": "Episode_Diagnostics/success_episode_count",
    "Episode_Diagnostics/start_object_x_mean_failure": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/start_object_y_mean_failure": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/failure_start_x_negative_offset_rate": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/failure_start_x_positive_offset_rate": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/failure_start_y_negative_offset_rate": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/failure_start_y_positive_offset_rate": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/arm_action_delta_rms_failure": "Episode_Diagnostics/failure_episode_count",
    "Episode_Diagnostics/gripper_switch_rate_failure": "Episode_Diagnostics/failure_episode_count",
}


def _to_float(value) -> float | None:
    """Convert scalar-like values from Isaac/RSL-RL logs to a Python float."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_log(info: dict) -> dict:
    """Find Isaac Lab episode log fields in the step info dict."""
    if not isinstance(info, dict):
        return {}
    log = info.get("log")
    return log if isinstance(log, dict) else {}


def _numeric_log(log: dict) -> dict[str, float]:
    """Convert a raw Isaac log dict to numeric values only."""
    numbers = {}
    for key, value in log.items():
        number = _to_float(value)
        if number is not None:
            numbers[key] = number
    return numbers


def _default_max_steps(env, requested_episodes: int, num_envs: int) -> int:
    """Give eval enough steps for requested episodes plus one extra timeout window."""
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    episode_steps = getattr(unwrapped, "max_episode_length", None)
    if episode_steps is None:
        step_dt = getattr(unwrapped, "step_dt", 0.02)
        episode_length_s = getattr(getattr(unwrapped, "cfg", None), "episode_length_s", 5.0)
        episode_steps = int(math.ceil(episode_length_s / step_dt))
    batches = int(math.ceil(requested_episodes / max(num_envs, 1)))
    return int((batches + 1) * episode_steps)


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve an explicit checkpoint or the configured run/checkpoint pair."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _make_runner(env, agent_cfg: RslRlBaseRunnerCfg, resume_path: str):
    """Create an RSL-RL runner and load the checkpoint."""
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    return runner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Evaluate an RSL-RL checkpoint and summarize episode logs."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = _resolve_checkpoint(agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = _make_runner(env, agent_cfg, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_model = runner.alg.policy
    except AttributeError:
        policy_model = runner.alg.actor_critic

    obs = env.get_observations()
    max_steps = args_cli.max_steps or _default_max_steps(env, args_cli.num_episodes, args_cli.num_envs)
    completed_episodes = 0
    weighted_totals: dict[str, float] = defaultdict(float)
    metric_weights: dict[str, float] = defaultdict(float)
    count_totals: dict[str, float] = defaultdict(float)
    logged_episode_weight = 0.0

    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, info = env.step(actions)
            policy_model.reset(dones)

            done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(sum(dones))
            log = _extract_log(info)
            numeric_log = _numeric_log(log)
            reset_count = numeric_log.get("Episode_Diagnostics/reset_env_count") if numeric_log else None
            log_weight = reset_count if reset_count is not None and reset_count > 0.0 else float(done_count)

            if numeric_log and log_weight > 0.0:
                logged_episode_weight += log_weight
                for key, number in numeric_log.items():
                    if key == "Episode_Diagnostics/reset_env_count":
                        continue
                    if key.endswith(COUNT_SUFFIX):
                        count_totals[key] += number
                        continue
                    if key in RATIO_SPECS:
                        continue
                    metric_weight_key = WEIGHT_KEY_BY_METRIC.get(key)
                    metric_weight = numeric_log.get(metric_weight_key, 0.0) if metric_weight_key else log_weight
                    if metric_weight > 0.0:
                        weighted_totals[key] += number * metric_weight
                        metric_weights[key] += metric_weight

            completed_episodes += done_count
            if completed_episodes >= args_cli.num_episodes:
                break

    metrics = {
        key: value / metric_weights[key]
        for key, value in sorted(weighted_totals.items())
        if metric_weights[key] > 0.0
    }
    metrics.update(dict(sorted(count_totals.items())))
    for rate_key, (numerator_key, denominator_key) in RATIO_SPECS.items():
        denominator = count_totals.get(denominator_key, 0.0)
        if denominator > 0.0:
            metrics[rate_key] = count_totals.get(numerator_key, 0.0) / denominator
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "requested_episodes": args_cli.num_episodes,
        "completed_episodes": completed_episodes,
        "logged_episode_weight": logged_episode_weight,
        "num_envs": args_cli.num_envs,
        "max_steps": max_steps,
        "metrics": metrics,
    }

    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(output + "\n")
        print(f"[INFO] Wrote eval summary: {args_cli.output}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
