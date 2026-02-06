# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Imitation learning tasks for Franka Factory.

These tasks are designed for data collection via teleoperation (CloudXR/Vision Pro)
with dual camera setup (wrist camera + table camera) for visuomotor policy learning.

Available environments:
    Teleoperation (data collection):
    - Franka-Factory-PegInsert-Teleop-v0
    - Franka-Factory-GearMesh-Teleop-v0
    - Franka-Factory-NutThread-Teleop-v0
    - Franka-Factory-MCXCardBlockInsert-Teleop-v0

    Mimic (data augmentation):
    - Franka-Factory-MCXCardBlockInsert-Mimic-v0
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments for imitation learning / teleoperation
##

gym.register(
    id="Franka-Factory-PegInsert-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_teleop_env_cfg:FactoryPegInsertTeleopEnvCfg",
    },
)

gym.register(
    id="Franka-Factory-GearMesh-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_teleop_env_cfg:FactoryGearMeshTeleopEnvCfg",
    },
)

gym.register(
    id="Franka-Factory-NutThread-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_teleop_env_cfg:FactoryNutThreadTeleopEnvCfg",
    },
)

gym.register(
    id="Franka-Factory-MCXCardBlockInsert-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.factory_teleop_env_cfg:MCXCardBlockInsertTeleopEnvCfg",
    },
)

# Mimic environment for data augmentation
gym.register(
    id="Franka-Factory-MCXCardBlockInsert-Mimic-v0",
    entry_point=f"{__name__}.mcx_block_mimic_env:MCXCardBlockInsertMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mcx_block_mimic_env_cfg:MCXCardBlockInsertMimicEnvCfg",
    },
)
