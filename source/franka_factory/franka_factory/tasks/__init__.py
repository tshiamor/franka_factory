# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tasks module for Franka Factory extension."""

from isaaclab_tasks.utils import import_packages

# Blacklist packages that should not be imported
_BLACKLIST_PKGS = ["utils", ".mdp"]

# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
