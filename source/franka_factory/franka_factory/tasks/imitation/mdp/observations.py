# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for Franka Factory imitation learning tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def ee_frame_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector position in relative frame (relative to environment origin).

    Using relative coordinates ensures consistency with get_object_poses() for Mimic.
    """
    ee_frame = env.scene[asset_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    # Convert to relative coordinates by subtracting environment origin
    env_origins = env.scene.env_origins
    return ee_pos_w - env_origins


def ee_frame_quat(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> torch.Tensor:
    """End-effector orientation (quaternion) in world frame.

    Orientation is the same in world and relative frames (only position differs).
    """
    ee_frame = env.scene[asset_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def gripper_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gripper finger positions."""
    robot = env.scene[asset_cfg.name]
    # Get finger joint positions (last 2 joints for Franka)
    return robot.data.joint_pos[:, -2:]


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "rgb",
    normalize: bool = False,
) -> torch.Tensor:
    """Get image observation from camera sensor.

    Args:
        env: The environment instance.
        sensor_cfg: Configuration for the camera sensor.
        data_type: Type of data to retrieve ("rgb", "depth", etc.).
        normalize: Whether to normalize the image to [0, 1].

    Returns:
        Image tensor from the camera.
    """
    camera = env.scene[sensor_cfg.name]
    img = camera.data.output[data_type]

    if normalize and data_type == "rgb":
        img = img.float() / 255.0

    return img
