#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay recorded demonstrations for Franka Factory tasks.

This script loads an HDF5 dataset and replays the recorded episodes.

Usage:
    # Replay all demos
    ./isaaclab.sh -p scripts/teleop/replay_demos.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 --dataset_file ./demos/mcx_card_demos.hdf5

    # Replay specific episode
    ./isaaclab.sh -p scripts/teleop/replay_demos.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 --dataset_file ./demos/mcx_card_demos.hdf5 --episode 0

    # Replay at slower speed
    ./isaaclab.sh -p scripts/teleop/replay_demos.py --task Franka-Factory-MCXCardBlockInsert-Teleop-v0 --dataset_file ./demos/mcx_card_demos.hdf5 --speed 0.5
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Replay recorded demonstrations for Franka Factory tasks.")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset file")
parser.add_argument("--episode", type=int, default=-1, help="Episode to replay (-1 for all)")
parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (0.5 = half speed)")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after simulator launch."""

import gymnasium as gym
import h5py
import numpy as np
import time
import torch

# Import franka factory extension
import franka_factory  # noqa: F401


def main():
    """Main function for replaying demonstrations."""
    # Load the dataset
    print(f"\nLoading dataset: {args_cli.dataset_file}")

    with h5py.File(args_cli.dataset_file, "r") as f:
        # Print dataset info
        print("\n" + "=" * 60)
        print("DATASET INFO")
        print("=" * 60)

        if "data" in f:
            episodes = list(f["data"].keys())
            print(f"Number of episodes: {len(episodes)}")

            for ep_name in episodes[:5]:  # Show first 5 episodes
                ep = f["data"][ep_name]
                if "actions" in ep:
                    num_steps = len(ep["actions"])
                    print(f"  {ep_name}: {num_steps} steps")

            if len(episodes) > 5:
                print(f"  ... and {len(episodes) - 5} more episodes")
        else:
            print("Dataset structure:")
            def print_structure(name, obj):
                print(f"  {name}: {type(obj)}")
            f.visititems(print_structure)

        print("=" * 60 + "\n")

    # Parse environment configuration
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    # Disable terminations for replay
    env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Reset environment
    env.reset()

    # Load and replay episodes
    with h5py.File(args_cli.dataset_file, "r") as f:
        if "data" not in f:
            print("Error: No 'data' group found in dataset")
            env.close()
            return

        episodes = sorted(f["data"].keys())

        if args_cli.episode >= 0:
            if args_cli.episode >= len(episodes):
                print(f"Error: Episode {args_cli.episode} not found. Dataset has {len(episodes)} episodes.")
                env.close()
                return
            episodes = [episodes[args_cli.episode]]

        print(f"\nReplaying {len(episodes)} episode(s) at {args_cli.speed}x speed\n")

        for ep_idx, ep_name in enumerate(episodes):
            ep = f["data"][ep_name]

            if "actions" not in ep:
                print(f"Skipping {ep_name}: no actions found")
                continue

            actions = np.array(ep["actions"])
            num_steps = len(actions)

            print("=" * 60)
            print(f"REPLAYING: {ep_name} ({ep_idx + 1}/{len(episodes)})")
            print(f"Steps: {num_steps}")
            print("=" * 60)

            # Reset environment for this episode
            env.reset()

            # Calculate step delay based on speed
            step_dt = env.cfg.sim.dt * env.cfg.decimation
            step_delay = step_dt / args_cli.speed

            # Replay actions
            for step_idx, action in enumerate(actions):
                start_time = time.time()

                # Convert action to tensor
                action_tensor = torch.tensor(action, device=env.device).unsqueeze(0)

                # Step environment
                env.step(action_tensor)

                # Print progress every 50 steps
                if step_idx % 50 == 0:
                    print(f"  Step {step_idx}/{num_steps}")

                # Rate limiting for visualization
                elapsed = time.time() - start_time
                if elapsed < step_delay:
                    time.sleep(step_delay - elapsed)

                # Check if simulation is still running
                if not simulation_app.is_running():
                    break

            print(f"  Completed {ep_name}\n")

            # Pause between episodes
            if ep_idx < len(episodes) - 1:
                print("  Next episode in 2 seconds...")
                time.sleep(2.0)

            if not simulation_app.is_running():
                break

    # Cleanup
    env.close()
    print("\nReplay completed!")


if __name__ == "__main__":
    main()
    simulation_app.close()
