#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based replay of demonstrations - sets exact recorded states.

This script replays demonstrations by directly setting robot joint states
and object poses from the recorded data, rather than applying actions.
This ensures perfect visual replay matching the original recording.
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="State-based replay of demonstrations.")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset file")
parser.add_argument("--episode", type=int, default=0, help="Episode to replay")
parser.add_argument("--output_dir", type=str, default="./videos", help="Directory to save videos")
parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
parser.add_argument("--camera", type=str, default="table_cam", help="Camera to record")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import h5py
import numpy as np
import os
import torch

import franka_factory  # noqa: F401


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset: {args_cli.dataset_file}")
    with h5py.File(args_cli.dataset_file, "r") as f:
        episodes = sorted(f["data"].keys())
        ep_name = episodes[args_cli.episode]
        ep = f["data"][ep_name]

        # Load all states
        robot_joint_pos = np.array(ep["states"]["articulation"]["robot"]["joint_position"])
        robot_joint_vel = np.array(ep["states"]["articulation"]["robot"]["joint_velocity"])
        block_poses = np.array(ep["states"]["rigid_object"]["block"]["root_pose"])
        block_vels = np.array(ep["states"]["rigid_object"]["block"]["root_velocity"])

        num_steps = len(robot_joint_pos)
        print(f"Episode: {ep_name}, Steps: {num_steps}")

    # Create environment
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Initialize cameras
    for _ in range(10):
        env.sim.render()

    # Get references to assets - use dict access
    robot = env.scene["robot"]
    block = env.scene["block"]
    env_ids = torch.tensor([0], device=env.device, dtype=torch.int32)

    print(f"\nRobot: {type(robot)}")
    print(f"Block: {type(block)}")

    # Get camera - use dict access
    if args_cli.camera not in env.scene.keys():
        print(f"Camera '{args_cli.camera}' not found!")
        print(f"Available: {env.scene.keys()}")
        env.close()
        return
    camera = env.scene[args_cli.camera]

    # Setup video writer
    # Do one render to get frame size
    camera.update(env.physics_dt)
    if hasattr(camera.data, 'output') and 'rgb' in camera.data.output:
        test_frame = camera.data.output['rgb'][0].cpu().numpy()
    else:
        test_frame = camera.data.rgb[0].cpu().numpy()

    if test_frame.dtype == np.float32:
        test_frame = (test_frame * 255).astype(np.uint8)
    if test_frame.shape[-1] == 4:
        test_frame = test_frame[..., :3]

    h, w = test_frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    video_path = os.path.join(args_cli.output_dir, f"{ep_name}_statebased_{args_cli.camera}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, args_cli.fps, (w, h))

    print(f"\n=== Replaying {ep_name} (state-based) ===")

    # Replay by setting states directly
    for step_idx in range(num_steps):
        # Set robot joint states
        joint_pos = torch.tensor(robot_joint_pos[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0)
        joint_vel = torch.tensor(robot_joint_vel[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0)

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Set block state
        block_pose = torch.tensor(block_poses[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0)
        block_vel = torch.tensor(block_vels[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0)

        block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
        block.write_root_velocity_to_sim(block_vel, env_ids=env_ids)

        # Force physics/render update
        env.sim.step(render=True)

        # Update camera and capture
        camera.update(env.physics_dt)
        if hasattr(camera.data, 'output') and 'rgb' in camera.data.output:
            frame = camera.data.output['rgb'][0].cpu().numpy()
        else:
            frame = camera.data.rgb[0].cpu().numpy()

        if frame.dtype == np.float32:
            frame = (frame * 255).astype(np.uint8)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

        if step_idx % 50 == 0:
            block_z = block_poses[step_idx][2]
            print(f"  Step {step_idx}/{num_steps}, block z={block_z:.3f}")

        if not simulation_app.is_running():
            break

    video_writer.release()
    print(f"\nSaved: {video_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
