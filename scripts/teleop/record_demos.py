#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record demonstrations for Franka Factory tasks using teleoperation.

This script allows users to record demonstrations for factory manipulation tasks
using CloudXR/Vision Pro or other teleoperation devices.

Usage:
    # Record demos with keyboard (default)
    ./isaaclab.sh -p scripts/teleop/record_demos.py --task Franka-Factory-PegInsert-Teleop-v0

    # Record demos with CloudXR/Vision Pro hand tracking
    ./isaaclab.sh -p scripts/teleop/record_demos.py --task Franka-Factory-PegInsert-Teleop-v0 --teleop_device handtracking

    # Record to specific file
    ./isaaclab.sh -p scripts/teleop/record_demos.py --task Franka-Factory-PegInsert-Teleop-v0 --dataset_file ./datasets/peg_insert_demos.hdf5
"""

import argparse
import contextlib
import os
import time

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Franka Factory tasks.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (e.g., Franka-Factory-PegInsert-Teleop-v0)")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleoperation device: keyboard, spacemouse, gamepad, or handtracking (for CloudXR/Vision Pro)",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/franka_factory_demos.hdf5",
    help="File path to save recorded demonstrations",
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demos to record (0 for unlimited)")
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Steps with success condition for marking demo as successful",
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable XR for hand tracking
app_launcher_args = vars(args_cli)
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after simulator launch."""

import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

# Import franka factory extension
import franka_factory  # noqa: F401

logger = logging.getLogger(__name__)


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def main():
    """Main function for recording demonstrations."""
    # Setup output directory
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Parse environment configuration
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task.split(":")[-1]

    # Extract success termination for checking
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None

    # Remove camera configs if using XR
    if args_cli.xr and not args_cli.enable_cameras:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Configure recorder
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir if output_dir else "."
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Rate limiter
    rate_limiter = None if args_cli.xr else RateLimiter(args_cli.step_hz)

    # State variables
    current_demo_count = 0
    success_step_count = 0
    should_reset = False
    recording_active = not args_cli.xr

    # Callbacks
    def reset_callback():
        nonlocal should_reset
        should_reset = True
        print("Reset requested")

    def start_callback():
        nonlocal recording_active
        recording_active = True
        print("Recording started")

    def stop_callback():
        nonlocal recording_active
        recording_active = False
        print("Recording paused")

    callbacks = {
        "R": reset_callback,
        "START": start_callback,
        "STOP": stop_callback,
        "RESET": reset_callback,
    }

    # Create teleop device
    teleop_interface = None
    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    else:
        if args_cli.teleop_device.lower() == "keyboard":
            teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
        elif args_cli.teleop_device.lower() == "spacemouse":
            teleop_interface = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
        else:
            logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
            env.close()
            return

        for key, callback in callbacks.items():
            teleop_interface.add_callback(key, callback)

    # Reset environment
    env.sim.reset()
    env.reset()
    teleop_interface.reset()

    print("\n" + "=" * 60)
    print("Franka Factory Demo Recording")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Device: {args_cli.teleop_device}")
    print(f"Output: {args_cli.dataset_file}")
    print("\nControls:")
    print("  - Press 'R' to reset the environment")
    if args_cli.xr:
        print("  - Pinch to start/stop recording")
    print("=" * 60 + "\n")

    # Main loop
    with contextlib.suppress(KeyboardInterrupt):
        with torch.inference_mode():
            while simulation_app.is_running():
                # Get teleop command
                action = teleop_interface.advance()
                actions = action.repeat(env.num_envs, 1)

                # Step environment
                if recording_active:
                    obs = env.step(actions)

                    # Check success condition
                    if success_term is not None:
                        if bool(success_term.func(env, **success_term.params)[0]):
                            success_step_count += 1
                            if success_step_count >= args_cli.num_success_steps:
                                env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                                env.recorder_manager.set_success_to_episodes(
                                    [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                                )
                                env.recorder_manager.export_episodes([0])
                                print("Demo completed successfully!")
                                should_reset = True
                        else:
                            success_step_count = 0
                else:
                    env.sim.render()

                # Check demo count
                if env.recorder_manager.exported_successful_episode_count > current_demo_count:
                    current_demo_count = env.recorder_manager.exported_successful_episode_count
                    print(f"Total successful demos: {current_demo_count}")

                # Check if done
                if args_cli.num_demos > 0 and current_demo_count >= args_cli.num_demos:
                    print(f"\nRecorded {current_demo_count} demos. Exiting...")
                    break

                # Handle reset
                if should_reset:
                    env.sim.reset()
                    env.recorder_manager.reset()
                    env.reset()
                    success_step_count = 0
                    should_reset = False

                # Check if stopped
                if env.sim.is_stopped():
                    break

                # Rate limiting
                if rate_limiter:
                    rate_limiter.sleep(env)

    # Cleanup
    env.close()
    print(f"\nSession completed with {current_demo_count} successful demonstrations")
    print(f"Saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    main()
    simulation_app.close()
