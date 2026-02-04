# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions for Franka Factory imitation learning tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set robot joints to default pose on reset.

    Args:
        env: The environment instance.
        env_ids: Environment indices to reset.
        default_pose: Default joint positions.
        asset_cfg: Configuration for the robot asset.
    """
    robot = env.scene[asset_cfg.name]
    joint_pos = torch.tensor(default_pose, device=env.device).unsqueeze(0).expand(len(env_ids), -1)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
