#!/usr/bin/env python
"""Replay demonstrations using scene.reset_to() method for exact state restoration."""

import argparse
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay demonstrations with reset_to method.")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset file")
parser.add_argument("--episode", type=int, default=None, help="Episode to replay (default: all episodes)")
parser.add_argument("--all", action="store_true", help="Replay all episodes")
parser.add_argument("--output_dir", type=str, default="./videos", help="Directory to save videos")
parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
parser.add_argument("--camera", type=str, default="table_cam", help="Camera to record")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# Default to episode 0 if neither --episode nor --all specified
if args_cli.episode is None and not args_cli.all:
    args_cli.episode = 0

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import gymnasium as gym
import h5py
import numpy as np
import os
import torch

import franka_factory  # noqa: F401


def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()


def replay_episode(env, camera, ep_name, ep_data, output_dir, fps, env_ids):
    """Replay a single episode and save video."""
    # Load all states
    robot_joint_pos = np.array(ep_data["states"]["articulation"]["robot"]["joint_position"])
    robot_joint_vel = np.array(ep_data["states"]["articulation"]["robot"]["joint_velocity"])
    robot_root_pose = np.array(ep_data["states"]["articulation"]["robot"]["root_pose"])
    robot_root_vel = np.array(ep_data["states"]["articulation"]["robot"]["root_velocity"])
    block_poses = np.array(ep_data["states"]["rigid_object"]["block"]["root_pose"])
    block_vels = np.array(ep_data["states"]["rigid_object"]["block"]["root_velocity"])

    num_steps = len(robot_joint_pos)
    log(f"  Steps: {num_steps}, Block z range: [{block_poses[:, 2].min():.3f}, {block_poses[:, 2].max():.3f}]")

    # Get frame dimensions
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

    video_path = os.path.join(output_dir, f"{ep_name}_{args_cli.camera}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    # Replay by constructing state dict and using reset_to
    for step_idx in range(num_steps):
        state = {
            "articulation": {
                "robot": {
                    "root_pose": torch.tensor(robot_root_pose[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                    "root_velocity": torch.tensor(robot_root_vel[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                    "joint_position": torch.tensor(robot_joint_pos[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                    "joint_velocity": torch.tensor(robot_joint_vel[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                }
            },
            "rigid_object": {
                "block": {
                    "root_pose": torch.tensor(block_poses[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                    "root_velocity": torch.tensor(block_vels[step_idx], device=env.device, dtype=torch.float32).unsqueeze(0),
                }
            },
            "deformable_object": {},
            "gripper": {},
        }

        env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
        env.sim.render()

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

        if not simulation_app.is_running():
            break

    video_writer.release()
    return video_path


def main():
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # Load dataset to get episode list
    log(f"\nLoading dataset: {args_cli.dataset_file}")
    with h5py.File(args_cli.dataset_file, "r") as f:
        all_episodes = sorted(f["data"].keys())
        log(f"Total episodes in dataset: {len(all_episodes)}")

    # Determine which episodes to replay
    if args_cli.all:
        episodes_to_replay = list(range(len(all_episodes)))
        log(f"Replaying ALL {len(episodes_to_replay)} episodes")
    else:
        episodes_to_replay = [args_cli.episode]
        log(f"Replaying episode {args_cli.episode}")

    # Create environment
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Initialize cameras
    log("Warming up cameras...")
    for _ in range(10):
        env.sim.render()

    # Get camera
    if args_cli.camera not in env.scene.keys():
        log(f"Camera '{args_cli.camera}' not found!")
        log(f"Available: {env.scene.keys()}")
        env.close()
        return
    camera = env.scene[args_cli.camera]

    env_ids = torch.tensor([0], device=env.device, dtype=torch.int32)

    # Replay each episode
    saved_videos = []
    with h5py.File(args_cli.dataset_file, "r") as f:
        for i, ep_idx in enumerate(episodes_to_replay):
            ep_name = all_episodes[ep_idx]
            log(f"\n=== Replaying {ep_name} ({i+1}/{len(episodes_to_replay)}) ===")

            ep_data = f["data"][ep_name]
            video_path = replay_episode(env, camera, ep_name, ep_data, args_cli.output_dir, args_cli.fps, env_ids)
            saved_videos.append(video_path)
            log(f"  Saved: {video_path}")

            # Reset environment between episodes
            env.reset()
            for _ in range(5):
                env.sim.render()

            if not simulation_app.is_running():
                break

    log(f"\n=== Summary ===")
    log(f"Saved {len(saved_videos)} videos to {args_cli.output_dir}")
    for v in saved_videos:
        log(f"  - {os.path.basename(v)}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
