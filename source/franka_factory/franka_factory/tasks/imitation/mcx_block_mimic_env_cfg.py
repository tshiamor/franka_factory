# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic environment configuration for MCX Card Block Insert task."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from franka_factory.tasks.imitation.factory_teleop_env_cfg import MCXCardBlockInsertTeleopEnvCfg


@configclass
class MCXCardBlockInsertMimicEnvCfg(MCXCardBlockInsertTeleopEnvCfg, MimicEnvCfg):
    """
    Isaac Lab Mimic environment config for MCX Card Block Insert task.

    This configuration defines the subtask structure for data augmentation:
    1. grasp_block - Approach and grasp the blue block
    2. place_block - Move and place the block on target (final subtask)
    """

    def __post_init__(self):
        # Initialize parent classes
        super().__post_init__()

        # Configure data generation parameters
        self.datagen_config.name = "mcx_block_insert_mimic"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 500  # Generate 500 demos from 15 source
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 100
        self.datagen_config.seed = 42

        # Define subtask configurations
        # For pick-and-place, we have 2 subtasks:
        # 1. Grasp the block (relative to block frame)
        # 2. Place the block (relative to target/block frame) - final subtask
        subtask_configs = []

        # Subtask 1: Grasp the block
        subtask_configs.append(
            SubTaskConfig(
                # Manipulation relative to the block
                object_ref="block",
                # Signal name for subtask completion (grasp detected)
                subtask_term_signal="grasp_block",
                # Random offsets for segment boundaries during generation
                subtask_term_offset_range=(5, 15),
                # Selection strategy for choosing source demo segment
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                # Action noise during this subtask
                action_noise=0.05,
                # Interpolation steps to bridge to this subtask
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                # Descriptions for UI/debugging
                description="Approach and grasp the blue block",
                next_subtask_description="Move block to target and place",
            )
        )

        # Subtask 2: Place the block (final subtask)
        subtask_configs.append(
            SubTaskConfig(
                # For placing, still relative to block (in gripper)
                object_ref="block",
                # Final subtask - no termination signal needed
                subtask_term_signal=None,
                # No offset for final subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                # Action noise
                action_noise=0.05,
                # Interpolation
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place block on target",
            )
        )

        # Register subtasks for the robot end-effector
        self.subtask_configs["franka"] = subtask_configs
