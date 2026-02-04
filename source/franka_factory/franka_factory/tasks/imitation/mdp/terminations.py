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
    card_pos: list[float],
    threshold: float = 0.03,
) -> torch.Tensor:
    """Check if the block is placed in the card's hole.

    Args:
        env: The environment instance.
        block_cfg: Configuration for the block asset.
        card_pos: Target position [x, y, z] of the card's hole.
        threshold: Distance threshold for success (default 3cm).

    Returns:
        Boolean tensor indicating if block is in the hole for each environment.
    """
    # Get block position
    block = env.scene[block_cfg.name]
    block_pos = block.data.root_pos_w[:, :3]

    # Convert card position to tensor
    target_pos = torch.tensor(card_pos, device=env.device).unsqueeze(0)

    # Calculate distance to target
    distance = torch.norm(block_pos - target_pos, dim=-1)

    # Task complete if block is within threshold of card hole
    return distance < threshold
