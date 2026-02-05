# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory teleoperation environment configuration for Franka robot.

This module provides manager-based environment configurations for factory tasks
with visuomotor observations (dual camera setup) and CloudXR/Vision Pro teleoperation
support for imitation learning data collection.

Camera setup:
    - wrist_cam: Mounted on end-effector (panda_hand) for close-up manipulation view
    - table_cam: Fixed overhead/side view for context awareness
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DeviceBase, DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.devices.openxr.openxr_device import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters.manipulator.gripper_retargeter import GripperRetargeterCfg
from isaaclab.devices.openxr.retargeters.manipulator.se3_rel_retargeter import Se3RelRetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Factory assets directory
FACTORY_ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

from . import mdp


##
# Scene Configuration
##
@configclass
class FactoryTeleopSceneCfg(InteractiveSceneCfg):
    """Scene configuration for factory teleoperation tasks.

    Includes:
        - Franka Panda robot
        - Table surface
        - Ground plane
        - Dual camera setup (wrist + table)
        - Lighting
    """

    # Robot - will be set in derived configs
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            semantic_tags=[("class", "table")],
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP Configuration
##
@configclass
class ActionsCfg:
    """Action specifications for teleoperation.

    Uses differential IK for smooth end-effector control.
    """

    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
    )

    gripper_action: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation specifications with visuomotor observations.

    Includes:
        - Proprioceptive state (joint positions, velocities, etc.)
        - RGB images from wrist and table cameras
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations with state and image data."""

        # State observations
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        # Image observations from cameras
        wrist_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
        )
        table_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Subtask success observations for task segmentation."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class EventCfg:
    """Event configuration for resetting the environment."""

    init_robot_pose = EventTerm(
        func=mdp.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400],
        },
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Base Environment Configuration
##
@configclass
class FactoryTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for factory teleoperation environments.

    This configuration provides:
        - Dual camera setup for visuomotor learning
        - CloudXR/Vision Pro teleoperation support
        - IK-based control for smooth end-effector manipulation
    """

    # Scene settings
    scene: FactoryTeleopSceneCfg = FactoryTeleopSceneCfg(num_envs=1, env_spacing=2.5)

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # XR/CloudXR configuration for Vision Pro teleoperation
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, -0.5, -1.05),  # Position relative to robot
        anchor_rot=(0.866, 0, 0, -0.5),  # Looking down at table
    )

    # List of image observations for visuomotor policy
    image_obs_list = ["wrist_cam", "table_cam"]

    def __post_init__(self):
        """Post initialization."""
        # General settings
        self.decimation = 5
        self.episode_length_s = 60.0  # Longer episodes for teleoperation

        # Simulation settings
        self.sim.dt = 0.01  # 100Hz physics
        self.sim.render_interval = 2

        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Set robot semantic tags
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Configure wrist camera (mounted on end-effector)
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.13, 0.0, -0.15),  # Forward and down from hand
                rot=(-0.70614, 0.03701, 0.03701, -0.70614),  # Looking forward
                convention="ros",
            ),
        )

        # Configure table camera (fixed overhead/side view)
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/table_cam",
            update_period=0.0,
            height=200,
            width=200,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 2),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.0, 0.4),  # Side view position
                rot=(0.35355, -0.61237, -0.61237, 0.35355),  # Looking at workspace
                convention="ros",
            ),
        )

        # Rendering settings for quality
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"

        # Teleoperation devices configuration (keyboard + CloudXR/Vision Pro)
        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        Se3RelRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            zero_out_xy_rotation=True,
                            use_wrist_rotation=False,
                            use_wrist_position=True,
                            delta_pos_scale_factor=10.0,
                            delta_rot_scale_factor=10.0,
                            sim_device=self.sim.device,
                        ),
                        GripperRetargeterCfg(
                            bound_hand=DeviceBase.TrackingTarget.HAND_RIGHT,
                            sim_device=self.sim.device,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )


##
# Task-Specific Configurations
##
@configclass
class FactoryPegInsertTeleopEnvCfg(FactoryTeleopEnvCfg):
    """Peg insertion task configuration for teleoperation.

    Uses the actual factory peg (8mm) and hole assets.
    """

    def __post_init__(self):
        super().__post_init__()

        # Common rigid body properties for factory assets
        factory_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        )

        # Add peg (held asset - yellow peg)
        self.scene.peg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Peg",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.15, 0.06], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_peg_8mm.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "peg")],
            ),
        )

        # Add hole/socket (fixed asset - grey socket, on table)
        # Using ArticulationCfg like reference factory tasks for proper USD loading
        self.scene.hole = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Hole",
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_hole_8mm.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.5, 0.15, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
            ),
            actuators={},
        )


@configclass
class FactoryGearMeshTeleopEnvCfg(FactoryTeleopEnvCfg):
    """Gear meshing task configuration for teleoperation.

    Uses the actual factory gear base and medium gear assets.
    """

    def __post_init__(self):
        super().__post_init__()

        # Common rigid body properties for factory assets
        factory_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        )

        # Add medium gear (held asset)
        self.scene.gear = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Gear",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.1, 0.06], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_gear_medium.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "gear")],
            ),
        )

        # Add gear base (fixed asset with pegs, elevated on table)
        self.scene.gear_base = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/GearBase",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.08], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_gear_base.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "gear_base")],
            ),
        )


@configclass
class FactoryNutThreadTeleopEnvCfg(FactoryTeleopEnvCfg):
    """Nut threading task configuration for teleoperation.

    Uses the actual factory nut (M16) and bolt assets.
    """

    def __post_init__(self):
        super().__post_init__()

        # Common rigid body properties for factory assets
        factory_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        )

        # Add nut (held asset)
        self.scene.nut = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Nut",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.1, 0.06], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_nut_m16.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "nut")],
            ),
        )

        # Add bolt (fixed asset, elevated on table)
        self.scene.bolt = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bolt",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.1, 0.08], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{FACTORY_ASSET_DIR}/factory_bolt_m16.usd",
                activate_contact_sensors=True,
                rigid_props=factory_rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                semantic_tags=[("class", "bolt")],
            ),
        )

        # Longer episode for threading task
        self.episode_length_s = 90.0


@configclass
class MCXCardBlockInsertTeleopEnvCfg(FactoryTeleopEnvCfg):
    """MCX Card block insertion task configuration for teleoperation.

    Uses MCX416A-BCAT network card as the slot and a blue block for insertion.
    The MCX card is fixed on the table facing upwards.
    The blue block spawns at random positions for pickup.
    """

    def __post_init__(self):
        super().__post_init__()

        # Common rigid body properties
        rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        )

        # Add blue block (held asset - to be inserted)
        # Scaled to cuboid: 10cm x 2cm x 1cm (0.1m x 0.02m x 0.01m)
        # Blue block default is ~5cm cube, so scale factors: (2.0, 0.4, 0.2)
        # Rotated 90° CCW about Y axis to stand on its base
        self.scene.block = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Block",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, -0.15, 0.03], rot=[0.707, 0, 0.707, 0]),
            spawn=UsdFileCfg(
                usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Blocks/blue_block.usd",
                scale=(2.0, 0.4, 0.2),  # Scale to 10cm x 2cm x 1cm cuboid
                activate_contact_sensors=True,
                rigid_props=rigid_props,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
                semantic_tags=[("class", "block")],
            ),
        )

        # MCX Card spawn configuration (shared by all cards)
        # Added collision properties so objects cannot pass through the cards
        mcx_card_spawn = UsdFileCfg(
            usd_path="/home/tshiamo/3D_Assets/3d model/cards/ImageToStl.com_MCX416A-BCAT_A601/MCX416A-BCAT_A601.usdc",
            scale=(0.001, 0.001, 0.001),  # Scale from mm to meters
            semantic_tags=[("class", "mcx_card")],
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        )

        # Add 5 MCX Cards arranged in a line along Y axis, spaced 5cm apart
        # Card 1 (primary target for block insertion)
        self.scene.mcx_card = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MCXCard",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.5, 0.15, 0.13],
                rot=[0.707, 0, 0.707, 0],  # 90° CCW about Y axis
            ),
            spawn=mcx_card_spawn,
        )

        # Card 2
        self.scene.mcx_card_2 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MCXCard_2",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.5, 0.20, 0.13],
                rot=[0.707, 0, 0.707, 0],
            ),
            spawn=mcx_card_spawn,
        )

        # Card 3
        self.scene.mcx_card_3 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MCXCard_3",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.5, 0.25, 0.13],
                rot=[0.707, 0, 0.707, 0],
            ),
            spawn=mcx_card_spawn,
        )

        # Card 4
        self.scene.mcx_card_4 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MCXCard_4",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.5, 0.30, 0.13],
                rot=[0.707, 0, 0.707, 0],
            ),
            spawn=mcx_card_spawn,
        )

        # Card 5
        self.scene.mcx_card_5 = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/MCXCard_5",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.5, 0.35, 0.13],
                rot=[0.707, 0, 0.707, 0],
            ),
            spawn=mcx_card_spawn,
        )

        # Target position for block placement
        # Position in front of the first MCX card where block should be placed
        target_pos = [0.45, 0.15, 0.08]  # In front of card 1, at table level + block half-height

        # VISIBLE TARGET MARKER - Green semi-transparent cube showing where to place block
        # This helps the user see exactly where to drop the block
        self.scene.target_marker = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TargetMarker",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[target_pos[0], target_pos[1], target_pos[2] + 0.05],  # Slightly above platform
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.12, 0.04, 0.10),  # Match blue block dimensions (10cm x 2cm x 1cm but taller for visibility)
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),  # No collision - visual only
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),  # Bright green
                    opacity=0.4,  # Semi-transparent so you can see through it
                ),
            ),
        )

        # PHYSICAL PLATFORM - Catches the block when dropped
        # This prevents the block from falling through
        self.scene.target_platform = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/TargetPlatform",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[target_pos[0], target_pos[1], target_pos[2] - 0.01],  # Just below target
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.15, 0.08, 0.02),  # 15cm x 8cm x 2cm platform
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),  # Has collision!
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.3),  # Dark gray
                    opacity=0.6,  # Slightly visible
                ),
            ),
        )

        # Add event to randomize block position on reset
        from isaaclab.envs.mdp import events as mdp_events

        self.events.reset_block_position = EventTerm(
            func=mdp_events.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.1, 0.1),   # Random X offset from default position
                    "y": (-0.1, 0.1),   # Random Y offset from default position
                    "z": (0.0, 0.02),   # Small Z variation
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("block"),
            },
        )

        # Socket position matches target platform
        socket_pos = [target_pos[0], target_pos[1], target_pos[2]]

        # Success condition: block placed on target platform
        # Block center should be within tolerance of socket position when resting on platform
        # NOTE: Named "success" so record_demos.py detects it properly
        self.terminations.success = DoneTerm(
            func=mdp.block_in_card_hole,
            params={
                "block_cfg": SceneEntityCfg("block"),
                "target_pos": [socket_pos[0], socket_pos[1], socket_pos[2] + 0.05],  # Block center when on platform
                "tolerance": [0.04, 0.03, 0.03],  # 4cm X, 3cm Y, 3cm Z - must be on/near the green marker
                "debug": True,  # Enable debug output
            },
        )

        # Episode length for block insertion
        self.episode_length_s = 60.0
