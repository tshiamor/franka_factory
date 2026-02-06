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
    debug: bool = True,
) -> torch.Tensor:
    """Check if the block is placed in the card's slot/hole position.

    Uses per-axis tolerance checking to verify the block is aligned with
    the target slot position in X, Y, and Z.

    Args:
        env: The environment instance.
        block_cfg: Configuration for the block asset.
        target_pos: Target position [x, y, z] of the slot/hole (relative to env origin).
        tolerance: Tolerance for each axis. Can be a single float (same for all axes)
                   or a list [x_tol, y_tol, z_tol] for per-axis tolerance.
        debug: If True, print block position periodically for debugging.

    Returns:
        Boolean tensor indicating if block is in the slot for each environment.
    """
    # Get block position in relative coordinates (relative to each environment's origin)
    block = env.scene[block_cfg.name]
    # Use world position minus environment origins to get relative position
    env_origins = env.scene.env_origins
    block_pos_rel = block.data.root_pos_w[:, :3] - env_origins[:, :3]

    # Convert target position to tensor (already in relative coords)
    target = torch.tensor(target_pos, device=env.device).unsqueeze(0)

    # Handle tolerance - convert to per-axis if single value
    if isinstance(tolerance, (int, float)):
        tol = torch.tensor([tolerance, tolerance, tolerance], device=env.device)
    else:
        tol = torch.tensor(tolerance, device=env.device)

    # Check if block is within tolerance on each axis (using relative position)
    diff = torch.abs(block_pos_rel - target)
    x_aligned = diff[:, 0] < tol[0]
    y_aligned = diff[:, 1] < tol[1]
    z_aligned = diff[:, 2] < tol[2]

    # Debug output every ~100 steps
    if debug and hasattr(env, '_term_debug_counter'):
        env._term_debug_counter += 1
        if env._term_debug_counter % 100 == 0:
            print(f"[DEBUG] Block pos (rel): [{block_pos_rel[0,0]:.3f}, {block_pos_rel[0,1]:.3f}, {block_pos_rel[0,2]:.3f}] | "
                  f"Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] | "
                  f"Diff: [{diff[0,0]:.3f}, {diff[0,1]:.3f}, {diff[0,2]:.3f}]")
    elif debug and not hasattr(env, '_term_debug_counter'):
        env._term_debug_counter = 0

    # Task complete if aligned on all axes
    success = x_aligned & y_aligned & z_aligned

    # Only print success message once (when first entering the target zone)
    was_in_target = getattr(env, '_block_was_in_target', False)
    if success.any() and not was_in_target:
        env._block_was_in_target = True
        print("\n" + "*" * 60)
        print("*" + " " * 20 + "BLOCK IN TARGET!" + " " * 20 + "*")
        print(f"*    Position (rel): [{block_pos_rel[0,0]:.3f}, {block_pos_rel[0,1]:.3f}, {block_pos_rel[0,2]:.3f}]")
        print("*" + " " * 58 + "*")
        print("*" * 60 + "\n")
    elif not success.any():
        env._block_was_in_target = False

    return success
