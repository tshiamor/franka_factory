#!/usr/bin/env python
# Copyright (c) 2024-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Render augmented demos with full camera modalities.

This script replays Mimic-generated demos and captures:
- RGB, Depth, Normals, Semantic/Instance Segmentation, Motion Vectors
- Robot state observations (eef_pos, eef_quat, gripper_pos, joint_pos, joint_vel)
- Actions

Usage:
    ./isaaclab.sh -p render_demos_full_modalities.py \
        --task Franka-Factory-MCXCardBlockInsert-Mimic-v0 \
        --input_file ./demos/mcx_card_demos_augmented.hdf5 \
        --output_file ./demos/mcx_card_demos_rendered.hdf5 \
        --resolution 84 \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Render demos with full modalities.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--input_file", type=str, required=True, help="Input augmented dataset.")
parser.add_argument("--output_file", type=str, required=True, help="Output rendered dataset.")
parser.add_argument("--resolution", type=int, default=84, help="Camera resolution (square).")
parser.add_argument("--start_episode", type=int, default=0, help="Start from this episode index.")
parser.add_argument("--max_episodes", type=int, default=None, help="Max episodes to render.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm

# Import franka_factory to register environments
import franka_factory  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def configure_cameras_for_full_modalities(env_cfg, resolution: int):
    """Configure cameras to output all modalities."""
    # Update camera resolution and data types
    if hasattr(env_cfg.scene, 'wrist_cam'):
        env_cfg.scene.wrist_cam.height = resolution
        env_cfg.scene.wrist_cam.width = resolution
        env_cfg.scene.wrist_cam.data_types = [
            "rgb",
            "depth",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "motion_vectors",
        ]

    if hasattr(env_cfg.scene, 'table_cam'):
        env_cfg.scene.table_cam.height = resolution
        env_cfg.scene.table_cam.width = resolution
        env_cfg.scene.table_cam.data_types = [
            "rgb",
            "depth",
            "normals",
            "semantic_segmentation",
            "instance_segmentation_fast",
            "motion_vectors",
        ]

    return env_cfg


def get_camera_data(env, camera_name: str) -> dict:
    """Get all modalities from a camera."""
    camera = env.scene[camera_name]
    data = {}

    # RGB (uint8, H x W x 3)
    if "rgb" in camera.data.output:
        rgb = camera.data.output["rgb"]
        if rgb is not None:
            # Remove alpha channel if present, convert to uint8
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            data["rgb"] = rgb[0].cpu().numpy().astype(np.uint8)

    # Depth (float32, H x W x 1)
    if "depth" in camera.data.output:
        depth = camera.data.output["depth"]
        if depth is not None:
            data["depth"] = depth[0].cpu().numpy().astype(np.float32)

    # Normals (float32, H x W x 3)
    if "normals" in camera.data.output:
        normals = camera.data.output["normals"]
        if normals is not None:
            # Normals are typically in range [-1, 1]
            data["normals"] = normals[0].cpu().numpy().astype(np.float32)

    # Semantic segmentation (uint32, H x W x 1)
    if "semantic_segmentation" in camera.data.output:
        semantic = camera.data.output["semantic_segmentation"]
        if semantic is not None:
            data["semantic"] = semantic[0].cpu().numpy().astype(np.uint32)

    # Instance segmentation (uint32, H x W x 1)
    if "instance_segmentation_fast" in camera.data.output:
        instance = camera.data.output["instance_segmentation_fast"]
        if instance is not None:
            data["instance"] = instance[0].cpu().numpy().astype(np.uint32)

    # Motion vectors / optical flow (float32, H x W x 2)
    if "motion_vectors" in camera.data.output:
        motion = camera.data.output["motion_vectors"]
        if motion is not None:
            data["motion_vectors"] = motion[0].cpu().numpy().astype(np.float32)

    return data


def get_robot_observations(env) -> dict:
    """Get robot state observations."""
    obs = {}

    # From observation buffer
    if "policy" in env.obs_buf:
        policy_obs = env.obs_buf["policy"]

        if "eef_pos" in policy_obs:
            obs["eef_pos"] = policy_obs["eef_pos"][0].cpu().numpy().astype(np.float32)

        if "eef_quat" in policy_obs:
            obs["eef_quat"] = policy_obs["eef_quat"][0].cpu().numpy().astype(np.float32)

        if "gripper_pos" in policy_obs:
            obs["gripper_pos"] = policy_obs["gripper_pos"][0].cpu().numpy().astype(np.float32)

        if "joint_pos" in policy_obs:
            obs["joint_pos"] = policy_obs["joint_pos"][0].cpu().numpy().astype(np.float32)

        if "joint_vel" in policy_obs:
            obs["joint_vel"] = policy_obs["joint_vel"][0].cpu().numpy().astype(np.float32)

    return obs


def replay_and_render_episode(
    env: ManagerBasedRLEnv,
    ep_data: dict,
    env_ids: torch.Tensor,
) -> dict:
    """Replay an episode and capture all modalities at each timestep."""

    # Get actions and states from the episode
    actions = ep_data["actions"]
    num_steps = len(actions)

    # Initialize storage for rendered data
    rendered_data = {
        "actions": [],
        "obs": {
            "eef_pos": [],
            "eef_quat": [],
            "gripper_pos": [],
            "joint_pos": [],
            "joint_vel": [],
            "wrist_rgb": [],
            "wrist_depth": [],
            "wrist_normals": [],
            "wrist_semantic": [],
            "wrist_instance": [],
            "wrist_motion_vectors": [],
            "table_rgb": [],
            "table_depth": [],
            "table_normals": [],
            "table_semantic": [],
            "table_instance": [],
            "table_motion_vectors": [],
        },
    }

    # Get initial state
    if "states" in ep_data:
        initial_state = {
            "articulation": {},
            "rigid_object": {},
            "deformable_object": {},
        }

        # Extract articulation states
        if "articulation" in ep_data["states"]:
            for asset_name, asset_data in ep_data["states"]["articulation"].items():
                initial_state["articulation"][asset_name] = {
                    "root_pose": torch.tensor(asset_data["root_pose"][0], device=env.device).unsqueeze(0),
                    "root_velocity": torch.tensor(asset_data["root_velocity"][0], device=env.device).unsqueeze(0),
                    "joint_position": torch.tensor(asset_data["joint_position"][0], device=env.device).unsqueeze(0),
                    "joint_velocity": torch.tensor(asset_data["joint_velocity"][0], device=env.device).unsqueeze(0),
                }

        # Extract rigid object states
        if "rigid_object" in ep_data["states"]:
            for obj_name, obj_data in ep_data["states"]["rigid_object"].items():
                initial_state["rigid_object"][obj_name] = {
                    "root_pose": torch.tensor(obj_data["root_pose"][0], device=env.device).unsqueeze(0),
                    "root_velocity": torch.tensor(obj_data["root_velocity"][0], device=env.device).unsqueeze(0),
                }

        # Reset to initial state
        env.scene.reset_to(initial_state, env_ids=env_ids, is_relative=True)
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

    # Replay each step
    for step_idx in range(num_steps):
        # Get state for this step
        if "states" in ep_data:
            step_state = {
                "articulation": {},
                "rigid_object": {},
                "deformable_object": {},
            }

            # Extract articulation states for this step
            if "articulation" in ep_data["states"]:
                for asset_name, asset_data in ep_data["states"]["articulation"].items():
                    step_state["articulation"][asset_name] = {
                        "root_pose": torch.tensor(asset_data["root_pose"][step_idx], device=env.device).unsqueeze(0),
                        "root_velocity": torch.tensor(asset_data["root_velocity"][step_idx], device=env.device).unsqueeze(0),
                        "joint_position": torch.tensor(asset_data["joint_position"][step_idx], device=env.device).unsqueeze(0),
                        "joint_velocity": torch.tensor(asset_data["joint_velocity"][step_idx], device=env.device).unsqueeze(0),
                    }

            # Extract rigid object states for this step
            if "rigid_object" in ep_data["states"]:
                for obj_name, obj_data in ep_data["states"]["rigid_object"].items():
                    step_state["rigid_object"][obj_name] = {
                        "root_pose": torch.tensor(obj_data["root_pose"][step_idx], device=env.device).unsqueeze(0),
                        "root_velocity": torch.tensor(obj_data["root_velocity"][step_idx], device=env.device).unsqueeze(0),
                    }

            # Set state
            env.scene.reset_to(step_state, env_ids=env_ids, is_relative=True)

        # Step simulation to update rendering
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

        # Render to get camera data
        env.sim.render()

        # Update observation buffer
        env.obs_buf = env.observation_manager.compute()

        # Store action
        rendered_data["actions"].append(actions[step_idx])

        # Get robot observations
        robot_obs = get_robot_observations(env)
        for key, value in robot_obs.items():
            if key in rendered_data["obs"]:
                rendered_data["obs"][key].append(value)

        # Get camera data for wrist camera
        if "wrist_cam" in env.scene.sensors:
            wrist_data = get_camera_data(env, "wrist_cam")
            if "rgb" in wrist_data:
                rendered_data["obs"]["wrist_rgb"].append(wrist_data["rgb"])
            if "depth" in wrist_data:
                rendered_data["obs"]["wrist_depth"].append(wrist_data["depth"])
            if "normals" in wrist_data:
                rendered_data["obs"]["wrist_normals"].append(wrist_data["normals"])
            if "semantic" in wrist_data:
                rendered_data["obs"]["wrist_semantic"].append(wrist_data["semantic"])
            if "instance" in wrist_data:
                rendered_data["obs"]["wrist_instance"].append(wrist_data["instance"])
            if "motion_vectors" in wrist_data:
                rendered_data["obs"]["wrist_motion_vectors"].append(wrist_data["motion_vectors"])

        # Get camera data for table camera
        if "table_cam" in env.scene.sensors:
            table_data = get_camera_data(env, "table_cam")
            if "rgb" in table_data:
                rendered_data["obs"]["table_rgb"].append(table_data["rgb"])
            if "depth" in table_data:
                rendered_data["obs"]["table_depth"].append(table_data["depth"])
            if "normals" in table_data:
                rendered_data["obs"]["table_normals"].append(table_data["normals"])
            if "semantic" in table_data:
                rendered_data["obs"]["table_semantic"].append(table_data["semantic"])
            if "instance" in table_data:
                rendered_data["obs"]["table_instance"].append(table_data["instance"])
            if "motion_vectors" in table_data:
                rendered_data["obs"]["table_motion_vectors"].append(table_data["motion_vectors"])

    # Convert lists to numpy arrays
    rendered_data["actions"] = np.array(rendered_data["actions"])
    for key in rendered_data["obs"]:
        if rendered_data["obs"][key]:
            rendered_data["obs"][key] = np.array(rendered_data["obs"][key])
        else:
            del rendered_data["obs"][key]

    return rendered_data


def load_episode_data(f: h5py.File, ep_name: str) -> dict:
    """Load episode data from HDF5 file."""
    ep_group = f["data"][ep_name]

    data = {
        "actions": np.array(ep_group["actions"]),
    }

    # Load states if available
    if "states" in ep_group:
        data["states"] = {}
        states_group = ep_group["states"]

        if "articulation" in states_group:
            data["states"]["articulation"] = {}
            for asset_name in states_group["articulation"].keys():
                asset_group = states_group["articulation"][asset_name]
                data["states"]["articulation"][asset_name] = {
                    key: np.array(asset_group[key]) for key in asset_group.keys()
                }

        if "rigid_object" in states_group:
            data["states"]["rigid_object"] = {}
            for obj_name in states_group["rigid_object"].keys():
                obj_group = states_group["rigid_object"][obj_name]
                data["states"]["rigid_object"][obj_name] = {
                    key: np.array(obj_group[key]) for key in obj_group.keys()
                }

    return data


def save_episode_data(f: h5py.File, ep_name: str, data: dict):
    """Save rendered episode data to HDF5 file."""
    if ep_name in f["data"]:
        del f["data"][ep_name]

    ep_group = f["data"].create_group(ep_name)

    # Save actions
    ep_group.create_dataset("actions", data=data["actions"], compression="gzip")

    # Save observations
    obs_group = ep_group.create_group("obs")
    for key, value in data["obs"].items():
        if isinstance(value, np.ndarray) and value.size > 0:
            obs_group.create_dataset(key, data=value, compression="gzip")


def main():
    """Render augmented demos with full modalities."""

    # Check input file exists
    if not os.path.exists(args_cli.input_file):
        raise FileNotFoundError(f"Input file not found: {args_cli.input_file}")

    # Create output directory
    output_dir = os.path.dirname(args_cli.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load input dataset info
    with h5py.File(args_cli.input_file, "r") as f:
        episode_names = list(f["data"].keys())
        env_name = f["data"].attrs.get("env_name", args_cli.task)

    total_episodes = len(episode_names)
    print(f"Input dataset: {args_cli.input_file}")
    print(f"Total episodes: {total_episodes}")
    print(f"Environment: {env_name}")

    # Apply episode range
    start_idx = args_cli.start_episode
    end_idx = total_episodes if args_cli.max_episodes is None else min(start_idx + args_cli.max_episodes, total_episodes)
    episode_names = episode_names[start_idx:end_idx]
    print(f"Rendering episodes {start_idx} to {end_idx - 1} ({len(episode_names)} episodes)")

    # Parse environment config
    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=1)

    # Configure cameras for full modalities
    env_cfg = configure_cameras_for_full_modalities(env_cfg, args_cli.resolution)

    # Disable terminations for replay
    env_cfg.terminations = None

    # Create environment
    print("Creating environment...")
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Environment IDs tensor
    env_ids = torch.tensor([0], device=env.device, dtype=torch.int32)

    # Create/open output file
    print(f"Output file: {args_cli.output_file}")

    with h5py.File(args_cli.output_file, "a") as f_out:
        # Create data group if not exists
        if "data" not in f_out:
            f_out.create_group("data")

        # Copy attributes
        f_out["data"].attrs["env_name"] = env_name
        f_out["data"].attrs["resolution"] = args_cli.resolution
        f_out["data"].attrs["modalities"] = [
            "rgb", "depth", "normals", "semantic", "instance", "motion_vectors"
        ]

        # Process each episode
        with h5py.File(args_cli.input_file, "r") as f_in:
            for ep_idx, ep_name in enumerate(tqdm(episode_names, desc="Rendering episodes")):
                try:
                    # Load episode data
                    ep_data = load_episode_data(f_in, ep_name)

                    # Replay and render
                    rendered_data = replay_and_render_episode(env, ep_data, env_ids)

                    # Save rendered data
                    save_episode_data(f_out, ep_name, rendered_data)

                    # Flush periodically
                    if (ep_idx + 1) % 10 == 0:
                        f_out.flush()

                except Exception as e:
                    print(f"\nError rendering {ep_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\nRendering complete!")
    print(f"Output saved to: {args_cli.output_file}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
