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
from eval_metrics import (  # isort: skip
    get_new_episode_log_weight,
    summarize_distribution,
    validate_exclusive_end_counts,
    validate_episode_log_total,
    validate_one_time_reward,
    validate_sequential_funnel,
)


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
import object_in_bowl  # noqa: F401
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

COUNT_SUFFIX = "_count"
SUM_SUFFIX = "_sum"
FUNNEL_COUNT_KEYS = [
    "Episode_Funnel/verified_grasp_count",
    "Episode_Funnel/lift_count",
    "Episode_Funnel/broad_entry_count",
    "Episode_Funnel/centered_count",
    "Episode_Funnel/lowered_count",
    "Episode_Funnel/opened_count",
    "Episode_Funnel/released_count",
    "Episode_Funnel/supported_settled_count",
]
RATIO_SPECS = {
    "Episode_Funnel/verified_grasp_given_completed_rate": (
        "Episode_Funnel/verified_grasp_count",
        "Episode_End/completed_count",
    ),
    "Episode_Funnel/lift_given_verified_grasp_rate": (
        "Episode_Funnel/lift_count",
        "Episode_Funnel/verified_grasp_count",
    ),
    "Episode_Funnel/broad_entry_given_lift_rate": (
        "Episode_Funnel/broad_entry_count",
        "Episode_Funnel/lift_count",
    ),
    "Episode_Funnel/centered_given_broad_entry_rate": (
        "Episode_Funnel/centered_count",
        "Episode_Funnel/broad_entry_count",
    ),
    "Episode_Funnel/lowered_given_centered_rate": (
        "Episode_Funnel/lowered_count",
        "Episode_Funnel/centered_count",
    ),
    "Episode_Funnel/opened_given_lowered_rate": (
        "Episode_Funnel/opened_count",
        "Episode_Funnel/lowered_count",
    ),
    "Episode_Funnel/released_given_opened_rate": (
        "Episode_Funnel/released_count",
        "Episode_Funnel/opened_count",
    ),
    "Episode_Funnel/supported_settled_given_released_rate": (
        "Episode_Funnel/supported_settled_count",
        "Episode_Funnel/released_count",
    ),
    "Episode_Success/success_5_rate": (
        "Episode_Success/success_5_count",
        "Episode_End/completed_count",
    ),
    "Episode_Success/success_10_given_success_5_rate": (
        "Episode_Success/success_10_count",
        "Episode_Success/success_5_count",
    ),
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
    "Episode_Entry/post_entry_outside_rate": (
        "Episode_Entry/post_entry_outside_step_count",
        "Episode_Entry/post_entry_step_count",
    ),
    "Episode_Entry/post_entry_below_lift_rate": (
        "Episode_Entry/post_entry_below_lift_step_count",
        "Episode_Entry/post_entry_step_count",
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
    "Episode_Diagnostics/min_xy_distance_after_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/start_object_x_mean_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/start_object_y_mean_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/arm_action_delta_rms_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/gripper_switch_rate_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/terminal_object_z_lift_mean": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/max_object_z_lift_mean": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/lift_retention_episode_mean_lift": "Episode_Diagnostics/lift_episode_count",
    "Episode_Diagnostics/terminal_above_lift_threshold_rate_lift": (
        "Episode_Diagnostics/lift_episode_count"
    ),
    "Episode_Diagnostics/fell_below_lift_threshold_after_lift_rate_lift": (
        "Episode_Diagnostics/lift_episode_count"
    ),
    "Episode_Diagnostics/start_object_x_mean_no_lift": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/start_object_y_mean_no_lift": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/no_lift_start_x_negative_offset_rate": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/no_lift_start_x_positive_offset_rate": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/no_lift_start_y_negative_offset_rate": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/no_lift_start_y_positive_offset_rate": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/arm_action_delta_rms_no_lift": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Diagnostics/gripper_switch_rate_no_lift": "Episode_Diagnostics/no_lift_episode_count",
    "Episode_Entry/first_lifted_tight_entry_step_mean": "Episode_Entry/lifted_tight_entry_count",
    "Episode_Entry/pre_entry_ring_step_mean": "Episode_Entry/lifted_tight_entry_count",
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


def _successful_lift_height_samples(info: dict) -> dict[str, list[float]]:
    """Extract raw successful-episode height samples from the environment info."""
    if not isinstance(info, dict):
        return {}
    payload = info.get("successful_lift_height_samples")
    if not isinstance(payload, dict):
        return {}
    samples = {}
    for key in ("terminal_object_z", "max_object_z"):
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            samples[key] = value.detach().flatten().float().cpu().tolist()
    return samples


def _episode_samples(info: dict) -> dict[str, list[float]]:
    """Extract raw episode samples used for exact endpoint distributions."""
    if not isinstance(info, dict):
        return {}
    payload = info.get("episode_samples")
    if not isinstance(payload, dict):
        return {}
    return {
        key: value.detach().flatten().float().cpu().tolist()
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }


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
    sum_totals: dict[str, float] = defaultdict(float)
    logged_episode_weight = 0.0
    successful_height_samples: dict[str, list[float]] = {
        "terminal_object_z": [],
        "max_object_z": [],
    }
    episode_samples: dict[str, list[float]] = defaultdict(list)

    with torch.inference_mode():
        for _ in range(max_steps):
            actions = policy(obs)
            obs, _, dones, info = env.step(actions)
            policy_model.reset(dones)

            done_count = int(dones.sum().item()) if isinstance(dones, torch.Tensor) else int(sum(dones))
            log = _extract_log(info)
            numeric_log = _numeric_log(log)
            reset_count = numeric_log.get("Episode_Diagnostics/reset_env_count") if numeric_log else None
            try:
                log_weight = get_new_episode_log_weight(done_count, reset_count)
            except ValueError as exc:
                raise RuntimeError(f"Invalid episode log at completed episode {completed_episodes}: {exc}") from exc

            if numeric_log and log_weight > 0.0:
                logged_episode_weight += log_weight
                for key, number in numeric_log.items():
                    if key == "Episode_Diagnostics/reset_env_count":
                        continue
                    if key.endswith(COUNT_SUFFIX):
                        count_totals[key] += number
                        continue
                    if key.endswith(SUM_SUFFIX):
                        sum_totals[key] += number
                        continue
                    if key in RATIO_SPECS:
                        continue
                    metric_weight_key = WEIGHT_KEY_BY_METRIC.get(key)
                    metric_weight = numeric_log.get(metric_weight_key, 0.0) if metric_weight_key else log_weight
                    if metric_weight > 0.0:
                        weighted_totals[key] += number * metric_weight
                        metric_weights[key] += metric_weight

                height_samples = _successful_lift_height_samples(info)
                for key, values in height_samples.items():
                    successful_height_samples[key].extend(values)
                for key, values in _episode_samples(info).items():
                    episode_samples[key].extend(values)

            completed_episodes += done_count
            if completed_episodes >= args_cli.num_episodes:
                break

    try:
        validate_episode_log_total(completed_episodes, logged_episode_weight)
    except ValueError as exc:
        raise RuntimeError(f"Evaluation log mismatch: {exc}") from exc

    metrics = {
        key: value / metric_weights[key]
        for key, value in sorted(weighted_totals.items())
        if metric_weights[key] > 0.0
    }
    metrics.update(dict(sorted(count_totals.items())))
    metrics.update(dict(sorted(sum_totals.items())))
    for key, value in sum_totals.items():
        if key.startswith("Episode_Reward_Sum/") and completed_episodes > 0:
            metrics[key.removesuffix(SUM_SUFFIX) + "_mean_per_episode"] = value / completed_episodes
    for rate_key, (numerator_key, denominator_key) in RATIO_SPECS.items():
        denominator = count_totals.get(denominator_key, 0.0)
        if denominator > 0.0:
            metrics[rate_key] = count_totals.get(numerator_key, 0.0) / denominator
    end_completed_count = count_totals.get("Episode_End/completed_count", 0.0)
    if int(round(end_completed_count)) != completed_episodes:
        raise RuntimeError(
            f"end-reason logs cover {end_completed_count:g} episodes, but evaluator observed {completed_episodes}"
        )
    try:
        validate_exclusive_end_counts(
            completed_episodes,
            count_totals.get("Episode_End/success_count", 0.0),
            count_totals.get("Episode_End/timeout_count", 0.0),
            count_totals.get("Episode_End/drop_count", 0.0),
            count_totals.get("Episode_End/other_count", 0.0),
        )
        validate_sequential_funnel(
            [("Episode_End/completed_count", end_completed_count)]
            + [(key, count_totals.get(key, 0.0)) for key in FUNNEL_COUNT_KEYS]
        )
        validate_sequential_funnel(
            [
                ("Episode_End/completed_count", end_completed_count),
                ("Episode_Success/success_5_count", count_totals.get("Episode_Success/success_5_count", 0.0)),
                (
                    "Episode_Success/success_10_count",
                    count_totals.get("Episode_Success/success_10_count", 0.0),
                ),
            ]
        )
        validate_one_time_reward(
            completed_episodes,
            count_totals.get("Episode_Entry/lifted_tight_entry_count", 0.0),
            sum_totals.get("Episode_Reward_Sum/tight_target_entry_bonus_sum", 0.0),
            reward_per_event=24.0,
        )
    except ValueError as exc:
        raise RuntimeError(f"Evaluation diagnostics are inconsistent: {exc}") from exc
    if count_totals.get("Episode_Success/success_10_count", 0.0) != count_totals.get(
        "Episode_End/success_count", 0.0
    ):
        raise RuntimeError("10-step funnel success count does not match success termination count")

    lift_episode_count = int(round(count_totals.get("Episode_Diagnostics/lift_episode_count", 0.0)))
    for sample_name, values in successful_height_samples.items():
        if len(values) != lift_episode_count:
            raise RuntimeError(
                f"collected {len(values)} {sample_name} samples for {lift_episode_count} lifted episodes"
            )
        for statistic, value in summarize_distribution(values).items():
            metrics[f"Episode_Diagnostics/{sample_name}_lift_{statistic}"] = value
    for sample_name, values in episode_samples.items():
        expected_count = lift_episode_count if sample_name == "min_xy_distance_after_lift" else completed_episodes
        if len(values) != expected_count:
            raise RuntimeError(
                f"collected {len(values)} {sample_name} samples, expected {expected_count}"
            )
        for statistic, value in summarize_distribution(values).items():
            metrics[f"Episode_Distribution/{sample_name}_{statistic}"] = value
    summary = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "requested_episodes": args_cli.num_episodes,
        "completed_episodes": completed_episodes,
        "episode_overshoot_count": completed_episodes - args_cli.num_episodes,
        "logged_episode_weight": logged_episode_weight,
        "num_envs": args_cli.num_envs,
        "max_steps": max_steps,
        "seed": agent_cfg.seed,
        "step_dt": env.unwrapped.step_dt,
        "sim_dt": env.unwrapped.cfg.sim.dt,
        "decimation": env.unwrapped.cfg.decimation,
        "episode_length_s": env.unwrapped.cfg.episode_length_s,
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
