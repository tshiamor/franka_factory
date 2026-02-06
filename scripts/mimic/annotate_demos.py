#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Wrapper script to annotate demos for franka_factory Mimic environments.
This imports franka_factory before running the annotation.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Annotate demonstrations for franka_factory environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--input_file", type=str, default="./datasets/dataset.hdf5", help="File name of the dataset to be annotated."
)
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/dataset_annotated.hdf5",
    help="File name of the annotated output dataset file.",
)
parser.add_argument("--auto", action="store_true", default=False, help="Automatically annotate subtasks.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import os
import torch

# Import franka_factory to register environments
import franka_factory  # noqa: F401

from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


class PreStepDatagenInfoRecorder(RecorderTerm):
    """Recorder term that records the datagen info data in each step."""

    def record_pre_step(self):
        eef_pose_dict = {}
        for eef_name in self._env.cfg.subtask_configs.keys():
            eef_pose_dict[eef_name] = self._env.get_robot_eef_pose(eef_name=eef_name)

        datagen_info = {
            "object_pose": self._env.get_object_poses(),
            "eef_pose": eef_pose_dict,
            "target_eef_pose": self._env.action_to_target_eef_pose(self._env.action_manager.action),
        }
        return "obs/datagen_info", datagen_info


@configclass
class PreStepDatagenInfoRecorderCfg(RecorderTermCfg):
    """Configuration for the datagen info recorder term."""
    class_type: type[RecorderTerm] = PreStepDatagenInfoRecorder


class PreStepSubtaskTermsObservationsRecorder(RecorderTerm):
    """Recorder term that records the subtask completion observations in each step."""

    def record_pre_step(self):
        return "obs/datagen_info/subtask_term_signals", self._env.get_subtask_term_signals()


@configclass
class PreStepSubtaskTermsObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step subtask terms observation recorder term."""
    class_type: type[RecorderTerm] = PreStepSubtaskTermsObservationsRecorder


@configclass
class MimicRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Mimic specific recorder terms."""
    record_pre_step_datagen_info = PreStepDatagenInfoRecorderCfg()
    record_pre_step_subtask_term_signals = PreStepSubtaskTermsObservationsRecorderCfg()


def replay_episode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Replays an episode in the environment."""
    initial_state = episode.data["initial_state"]
    actions = episode.data["actions"]
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset_to(initial_state, None, is_relative=True)

    for action_index, action in enumerate(actions):
        action_tensor = torch.Tensor(action).reshape([1, action.shape[0]])
        env.step(torch.Tensor(action_tensor))

    if success_term is not None:
        if not bool(success_term.func(env, **success_term.params)[0]):
            return False
    return True


def annotate_episode_in_auto_mode(
    env: ManagerBasedRLMimicEnv,
    episode: EpisodeData,
    success_term: TerminationTermCfg | None = None,
) -> bool:
    """Annotates an episode in automatic mode."""
    is_episode_annotated_successfully = replay_episode(env, episode, success_term)

    if not is_episode_annotated_successfully:
        print("\tThe final task was not completed.")
    else:
        # check if all the subtask term signals are annotated
        annotated_episode = env.recorder_manager.get_episode(0)
        subtask_term_signal_dict = annotated_episode.data["obs"]["datagen_info"]["subtask_term_signals"]
        for signal_name, signal_flags in subtask_term_signal_dict.items():
            signal_flags = torch.tensor(signal_flags, device=env.device)
            if not torch.any(signal_flags):
                is_episode_annotated_successfully = False
                print(f'\tDid not detect completion for the subtask "{signal_name}".')

    return is_episode_annotated_successfully


def main():
    """Add Isaac Lab Mimic annotations to the given demo dataset file."""
    # Load input dataset to be annotated
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"The input dataset file {args_cli.input_file} does not exist.")

    dataset_file_handler = HDF5DatasetFileHandler()
    dataset_file_handler.open(args_cli.input_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        return 0

    # get output directory path and file name
    output_dir = os.path.dirname(args_cli.output_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args_cli.task is not None:
        env_name = args_cli.task.split(":")[-1]
    if env_name is None:
        raise ValueError("Task/env name was not specified nor found in the dataset.")

    print(f"Using environment: {env_name}")
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.env_name = env_name

    # extract success checking function
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    # Disable all termination terms
    env_cfg.terminations = None

    # Set up recorder terms for mimic annotations
    env_cfg.recorders = MimicRecorderManagerCfg()
    if not args_cli.auto:
        env_cfg.recorders.record_pre_step_subtask_term_signals = None

    env_cfg.recorders.dataset_export_dir_path = output_dir if output_dir else "."
    env_cfg.recorders.dataset_filename = output_file_name

    # create environment
    env: ManagerBasedRLMimicEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if args_cli.auto:
        # check if the mimic API is implemented
        if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
            raise NotImplementedError(
                "The environment does not implement the get_subtask_term_signals method required "
                "to run automatic annotations."
            )

    # reset environment
    env.reset()

    # simulate environment
    exported_episode_count = 0
    processed_episode_count = 0
    successful_task_count = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            for episode_index, episode_name in enumerate(dataset_file_handler.get_episode_names()):
                processed_episode_count += 1
                print(f"\nAnnotating episode #{episode_index} ({episode_name})")
                episode = dataset_file_handler.load_episode(episode_name, env.device)

                is_episode_annotated_successfully = False
                if args_cli.auto:
                    is_episode_annotated_successfully = annotate_episode_in_auto_mode(env, episode, success_term)
                else:
                    print("Manual annotation not implemented in this script. Use --auto flag.")
                    is_episode_annotated_successfully = False

                if is_episode_annotated_successfully:
                    env.recorder_manager.set_success_to_episodes(
                        None, torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes()
                    exported_episode_count += 1
                    successful_task_count += 1
                    print("\tExported the annotated episode.")
                else:
                    print("\tSkipped exporting the episode due to incomplete subtask annotations.")
            break

    print(f"\nExported {exported_episode_count} (out of {processed_episode_count}) annotated episodes.")
    print(f"Successful task completions: {successful_task_count}")
    print("Exiting the app.")

    env.close()
    return successful_task_count


if __name__ == "__main__":
    successful_task_count = main()
    simulation_app.close()
    exit(successful_task_count)
