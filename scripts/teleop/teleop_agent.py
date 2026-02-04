#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to teleoperate Franka Factory environments.

This script allows interactive teleoperation for testing and practice
before recording demonstrations.

Usage:
    # Teleoperate with keyboard
    ./isaaclab.sh -p scripts/teleop/teleop_agent.py --task Franka-Factory-PegInsert-Teleop-v0

    # Teleoperate with CloudXR/Vision Pro
    ./isaaclab.sh -p scripts/teleop/teleop_agent.py --task Franka-Factory-PegInsert-Teleop-v0 --teleop_device handtracking
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperate Franka Factory environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleoperation device: keyboard, spacemouse, gamepad, or handtracking",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Control sensitivity")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable XR for hand tracking
if "handtracking" in args_cli.teleop_device.lower():
    args_cli.xr = True

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after simulator launch."""

import gymnasium as gym
import logging
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg, Se3Gamepad, Se3GamepadCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device

# Import franka factory extension
import franka_factory  # noqa: F401

logger = logging.getLogger(__name__)


def main():
    """Main teleoperation loop."""
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    env_cfg.terminations.time_out = None

    # Remove cameras if using XR
    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # State variables
    should_reset = False
    teleop_active = not args_cli.xr

    # Callbacks
    def reset_callback():
        nonlocal should_reset
        should_reset = True
        print("Reset triggered")

    def start_callback():
        nonlocal teleop_active
        teleop_active = True
        print("Teleoperation activated")

    def stop_callback():
        nonlocal teleop_active
        teleop_active = False
        print("Teleoperation deactivated")

    callbacks = {
        "R": reset_callback,
        "START": start_callback,
        "STOP": stop_callback,
        "RESET": reset_callback,
    }

    # Create teleop device
    sensitivity = args_cli.sensitivity
    teleop_interface = None

    if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
        teleop_interface = create_teleop_device(args_cli.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    else:
        if args_cli.teleop_device.lower() == "keyboard":
            teleop_interface = Se3Keyboard(
                Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
        elif args_cli.teleop_device.lower() == "spacemouse":
            teleop_interface = Se3SpaceMouse(
                Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
        elif args_cli.teleop_device.lower() == "gamepad":
            teleop_interface = Se3Gamepad(
                Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
            )
        else:
            logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
            env.close()
            return

        for key, callback in callbacks.items():
            try:
                teleop_interface.add_callback(key, callback)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to add callback for key {key}: {e}")

    # Reset environment
    env.reset()
    teleop_interface.reset()

    print("\n" + "=" * 60)
    print("Franka Factory Teleoperation")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Device: {args_cli.teleop_device}")
    print("\nControls:")
    print("  - Press 'R' to reset the environment")
    if args_cli.teleop_device.lower() == "keyboard":
        print("  - WASD for X/Y movement")
        print("  - Q/E for Z movement")
        print("  - Arrow keys for rotation")
        print("  - Space for gripper toggle")
    print("=" * 60 + "\n")

    # Main loop
    while simulation_app.is_running():
        try:
            with torch.inference_mode():
                action = teleop_interface.advance()

                if teleop_active:
                    actions = action.repeat(env.num_envs, 1)
                    env.step(actions)
                else:
                    env.sim.render()

                if should_reset:
                    env.reset()
                    teleop_interface.reset()
                    should_reset = False
                    print("Environment reset complete")

        except Exception as e:
            logger.error(f"Error during simulation: {e}")
            break

    env.close()
    print("Teleoperation ended")


if __name__ == "__main__":
    main()
    simulation_app.close()
