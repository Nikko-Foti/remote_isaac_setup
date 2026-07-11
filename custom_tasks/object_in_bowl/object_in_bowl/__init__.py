# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-in-bowl Isaac Lab task package."""

import gymnasium as gym

from . import ppo_cfg

gym.register(
    id="Isaac-Object-In-Bowl-Franka-v0",
    entry_point=f"{__name__}.env:ObjectInBowlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:ObjectInBowlEnvCfg",
        "rsl_rl_cfg_entry_point": f"{ppo_cfg.__name__}:PPORunnerCfg",
    },
)

# Register the optional Omniverse UI extension.
from .ui_extension_example import *  # noqa: E402, F401, F403
