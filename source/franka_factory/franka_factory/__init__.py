# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Factory extension for IsaacLab.

This extension provides factory manipulation tasks using the Franka Panda robot.
"""

import os
import toml

# Conveniences to other module directories via relative paths
FRANKA_FACTORY_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the extension source directory."""

FRANKA_FACTORY_METADATA = toml.load(os.path.join(FRANKA_FACTORY_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = FRANKA_FACTORY_METADATA["package"]["version"]

##
# Register Gym environments.
##
from .tasks import *
