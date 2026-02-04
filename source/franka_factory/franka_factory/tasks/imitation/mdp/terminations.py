# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions for Franka Factory imitation learning tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def block_in_card_hole(
    env: ManagerBasedRLEnv,
    block_cfg: SceneEntityCfg,
    target_pos: list[float],
    tolerance: list[float] | float = 0.02,
) -> torch.Tensor:
    """Check if the block is placed in the card's slot/hole position.

    Uses per-axis tolerance checking to verify the block is aligned with
    the target slot position in X, Y, and Z.

    Args:
        env: The environment instance.
        block_cfg: Configuration for the block asset.
        target_pos: Target position [x, y, z] of the slot/hole.
        tolerance: Tolerance for each axis. Can be a single float (same for all axes)
                   or a list [x_tol, y_tol, z_tol] for per-axis tolerance.

    Returns:
        Boolean tensor indicating if block is in the slot for each environment.
    """
    # Get block position
    block = env.scene[block_cfg.name]
    block_pos = block.data.root_pos_w[:, :3]

    # Convert target position to tensor
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)

    # Handle tolerance - convert to per-axis if single value
    if isinstance(tolerance, (int, float)):
        tol = torch.tensor([tolerance, tolerance, tolerance], device=env.device)
    else:
        tol = torch.tensor(tolerance, device=env.device)

    # Check if block is within tolerance on each axis
    diff = torch.abs(block_pos - target)
    x_aligned = diff[:, 0] < tol[0]
    y_aligned = diff[:, 1] < tol[1]
    z_aligned = diff[:, 2] < tol[2]

    # Task complete if aligned on all axes
    return x_aligned & y_aligned & z_aligned
