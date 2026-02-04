#!/usr/bin/env python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to list all registered environments in the Franka Factory extension."""

import gymnasium as gym

# Import the extension to register environments
import franka_factory  # noqa: F401


def main():
    """List all Franka Factory environments."""
    print("\n" + "=" * 60)
    print("Franka Factory Environments")
    print("=" * 60 + "\n")

    # Get all registered environments
    all_envs = gym.envs.registry.keys()

    # Filter for Franka Factory environments
    franka_envs = [env for env in all_envs if "Franka-Factory" in env]

    if franka_envs:
        for env in sorted(franka_envs):
            print(f"  - {env}")
    else:
        print("  No Franka Factory environments found.")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
