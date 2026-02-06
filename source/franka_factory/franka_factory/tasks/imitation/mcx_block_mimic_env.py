# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mimic environment for MCX Card Block Insert task."""

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class MCXCardBlockInsertMimicEnv(ManagerBasedRLMimicEnv):
    """
    Isaac Lab Mimic environment wrapper for MCX Card Block Insert task.

    This environment enables data augmentation using the MimicGen approach.
    """

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        # Retrieve end effector pose from the observation buffer
        eef_pos = self.obs_buf["policy"]["eef_pos"][env_ids]
        eef_quat = self.obs_buf["policy"]["eef_quat"][env_ids]
        # Quaternion format is w,x,y,z
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    # Note: get_object_poses is inherited from base class ManagerBasedRLMimicEnv
    # It uses self.scene.get_state(is_relative=True) which returns all rigid objects
    # in relative coordinates (relative to each environment's origin)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """
        Converts target pose and gripper action to an environment action.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action.
            env_id: Environment index.

        Returns:
            An action torch.Tensor compatible with env.step().
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        # target position and rotation
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # normalized delta position action
        delta_position = target_pos - curr_pos

        # normalized delta rotation action
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        # get gripper action for single eef
        (gripper_action,) = gripper_action_dict.values()

        # add noise to action
        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action += noise
            pose_action = torch.clamp(pose_action, -1.0, 1.0)

        return torch.cat([pose_action, gripper_action], dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action to target pose for the end effector controller.
        Inverse of target_eef_pose_to_action.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor
        """
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        # current position and rotation
        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        # get pose target
        target_pos = curr_pos + delta_position

        # Convert delta_rotation to rotation matrix
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        # Handle near-zero angle
        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()

        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper action from environment actions.

        Args:
            actions: Environment actions. Shape is (num_envs, num_steps, action_dim).

        Returns:
            Dictionary of gripper actions.
        """
        # last dimension is gripper action
        return {list(self.cfg.subtask_configs.keys())[0]: actions[:, -1:]}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """
        Gets termination signals for each subtask.

        For MCXCardBlockInsert, we have one subtask boundary: grasp_block.
        The grasp is detected when the gripper is closed and the block is lifted.

        Args:
            env_ids: Environment indices. If None, all envs are considered.

        Returns:
            Dictionary of termination signals for each subtask.
        """
        if env_ids is None:
            env_ids = slice(None)

        signals = {}

        # Get block z-position in relative coordinates (relative to env origin)
        block = self.scene["block"]
        env_origins = self.scene.env_origins
        if isinstance(env_ids, slice):
            block_z = block.data.root_pos_w[:, 2] - env_origins[:, 2]
        else:
            block_z = block.data.root_pos_w[env_ids, 2] - env_origins[env_ids, 2]

        # Get gripper state
        gripper_pos = self.obs_buf["policy"]["gripper_pos"][env_ids]
        gripper_closed = gripper_pos[:, 0] < 0.02  # Fingers close together

        # Grasp is complete when block is lifted above threshold and gripper is closed
        block_lifted = block_z > 0.08  # Above table surface (relative z)
        grasp_complete = torch.logical_and(gripper_closed, block_lifted)

        signals["grasp_block"] = grasp_complete

        return signals

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """
        Return the expected attached object for the given subtask.

        For place subtask, we expect the block to be attached.
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        # If this is the place subtask (index 1), expect block attached
        if subtask_index == 1:
            return "block"

        return None
