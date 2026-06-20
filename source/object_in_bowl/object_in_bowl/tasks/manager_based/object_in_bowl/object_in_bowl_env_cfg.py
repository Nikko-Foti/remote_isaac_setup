# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


OBJECT_START_POSITION = (0.50, 0.0, 0.055)
OBJECT_TARGET_REWARD_MIN_HEIGHT = 0.04
OBJECT_OFFICIAL_LIFTED_HEIGHT = 0.04
PLACEMENT_TARGET_POSITION = (0.70, 0.20, 0.061)
PLACEMENT_COMMAND_X_RANGE = (0.62, 0.78)
PLACEMENT_COMMAND_Y_RANGE = (0.12, 0.28)
PLACEMENT_COMMAND_Z_RANGE = (0.25, 0.45)
OBJECT_LIFTED_HEIGHT = 0.105
PLACEMENT_TARGET_RADIUS = 0.08
BOWL_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd"
BOWL_ASSET_POSITION = (PLACEMENT_TARGET_POSITION[0], PLACEMENT_TARGET_POSITION[1], 0.025)
BOWL_ASSET_ROTATION = (0.7071068, -0.7071068, 0.0, 0.0)
BOWL_SUCCESS_RADIUS = 0.11
BOWL_COLLISION_INNER_HALF_SIZE = BOWL_SUCCESS_RADIUS
BOWL_COLLISION_WALL_THICKNESS = 0.02
BOWL_COLLISION_WALL_HEIGHT = 0.08
BOWL_COLLISION_BASE_THICKNESS = 0.012
BOWL_COLLISION_OUTER_SIZE = 2.0 * (BOWL_COLLISION_INNER_HALF_SIZE + BOWL_COLLISION_WALL_THICKNESS)
BOWL_COLLISION_WALL_CENTER_Z = PLACEMENT_TARGET_POSITION[2] + BOWL_COLLISION_WALL_HEIGHT / 2.0
BOWL_LOWERING_RADIUS = BOWL_SUCCESS_RADIUS
BOWL_SUCCESS_MIN_HEIGHT = PLACEMENT_TARGET_POSITION[2] - 0.005
BOWL_SUCCESS_MAX_HEIGHT = PLACEMENT_TARGET_POSITION[2] + 0.06
BOWL_LOWERING_TARGET_HEIGHT = (BOWL_SUCCESS_MIN_HEIGHT + BOWL_SUCCESS_MAX_HEIGHT) / 2.0
BOWL_SUCCESS_MAX_SPEED = 0.30
BOWL_SUCCESS_MAX_ANGULAR_SPEED = 1.0
BOWL_SUCCESS_MIN_GRIPPER_OPEN = 0.03


##
# Scene definition
##


# Builds the sensor that tells Isaac Lab where the Franka hand is.
def _make_ee_frame_cfg() -> FrameTransformerCfg:
    """Create the end-effector frame sensor without adding marker config as a scene entity."""
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
            ),
        ],
    )


# Defines what physical things exist in each copy of the environment.
@configclass
class ObjectInBowlSceneCfg(InteractiveSceneCfg):
    """Configuration for the first Franka object-in-bowl scene."""

    # robot
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # end-effector frame used by later observations and rewards
    ee_frame = _make_ee_frame_cfg()

    # table and ground plane follow the built-in Franka lift task layout
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # cube first; a ball can come later after the pick/place loop works
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=list(OBJECT_START_POSITION), rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    # fixed YCB bowl at the placement corner
    bowl = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bowl",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(BOWL_ASSET_POSITION), rot=list(BOWL_ASSET_ROTATION)),
        spawn=UsdFileCfg(usd_path=BOWL_USD_PATH),
    )

    # invisible simple collision tray for the visual bowl
    bowl_collision_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlCollisionBase",
        init_state=AssetBaseCfg.InitialStateCfg(pos=list(PLACEMENT_TARGET_POSITION)),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_BASE_THICKNESS),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9), opacity=0.0),
        ),
    )

    bowl_collision_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlCollisionFront",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                PLACEMENT_TARGET_POSITION[0],
                PLACEMENT_TARGET_POSITION[1] + BOWL_COLLISION_INNER_HALF_SIZE + BOWL_COLLISION_WALL_THICKNESS / 2.0,
                BOWL_COLLISION_WALL_CENTER_Z,
            ]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_WALL_THICKNESS, BOWL_COLLISION_WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9), opacity=0.0),
        ),
    )

    bowl_collision_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlCollisionBack",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                PLACEMENT_TARGET_POSITION[0],
                PLACEMENT_TARGET_POSITION[1] - BOWL_COLLISION_INNER_HALF_SIZE - BOWL_COLLISION_WALL_THICKNESS / 2.0,
                BOWL_COLLISION_WALL_CENTER_Z,
            ]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_WALL_THICKNESS, BOWL_COLLISION_WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9), opacity=0.0),
        ),
    )

    bowl_collision_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlCollisionLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                PLACEMENT_TARGET_POSITION[0] - BOWL_COLLISION_INNER_HALF_SIZE - BOWL_COLLISION_WALL_THICKNESS / 2.0,
                PLACEMENT_TARGET_POSITION[1],
                BOWL_COLLISION_WALL_CENTER_Z,
            ]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_WALL_THICKNESS, 2.0 * BOWL_COLLISION_INNER_HALF_SIZE, BOWL_COLLISION_WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9), opacity=0.0),
        ),
    )

    bowl_collision_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BowlCollisionRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                PLACEMENT_TARGET_POSITION[0] + BOWL_COLLISION_INNER_HALF_SIZE + BOWL_COLLISION_WALL_THICKNESS / 2.0,
                PLACEMENT_TARGET_POSITION[1],
                BOWL_COLLISION_WALL_CENTER_Z,
            ]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_WALL_THICKNESS, 2.0 * BOWL_COLLISION_INNER_HALF_SIZE, BOWL_COLLISION_WALL_HEIGHT),
            collision_props=CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 0.9), opacity=0.0),
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


# Defines the sampled carry target above the visual bowl area.
@configclass
class CommandsCfg:
    """Command terms that ask the policy to carry the cube above the bowl area."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=PLACEMENT_COMMAND_X_RANGE,
            pos_y=PLACEMENT_COMMAND_Y_RANGE,
            pos_z=PLACEMENT_COMMAND_Z_RANGE,
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


# Defines what actions the policy can send to the robot.
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


# Defines what information the policy gets to observe.
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # This is the observation group used by the learning policy.
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.get_object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        # Tells Isaac Lab to combine these observations into one policy input vector.
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# Defines what gets reset at the start of each episode.
@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


# Defines the lift, carry, and bowl-placement rewards.
@configclass
class RewardsCfg:
    """Reward terms for lifting the cube and placing it in the bowl."""

    reaching_object = RewTerm(
        func=mdp.compute_reaching_object_reward,
        weight=1.0,
        params={"std": 0.10},
    )
    lifting_object = RewTerm(
        func=mdp.compute_object_lifted_reward,
        weight=15.0,
        params={"minimal_height": OBJECT_OFFICIAL_LIFTED_HEIGHT},
    )
    object_goal_tracking = RewTerm(
        func=mdp.compute_object_goal_distance_reward,
        weight=16.0,
        params={
            "std": 0.30,
            "minimal_height": OBJECT_TARGET_REWARD_MIN_HEIGHT,
            "command_name": "object_pose",
            "gate_target_position": PLACEMENT_TARGET_POSITION,
            "gate_radius": BOWL_LOWERING_RADIUS,
            "gate_minimal_height": BOWL_SUCCESS_MIN_HEIGHT,
            "gate_reward_scale": 0.5,
        },
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.compute_object_goal_distance_reward,
        weight=5.0,
        params={
            "std": 0.05,
            "minimal_height": OBJECT_TARGET_REWARD_MIN_HEIGHT,
            "command_name": "object_pose",
            "gate_target_position": PLACEMENT_TARGET_POSITION,
            "gate_radius": BOWL_LOWERING_RADIUS,
            "gate_minimal_height": BOWL_SUCCESS_MIN_HEIGHT,
            "gate_reward_scale": 0.0,
        },
    )
    object_to_target_xy = RewTerm(
        func=mdp.compute_object_to_target_xy_reward,
        weight=0.0,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "std": 0.20,
            "radius": PLACEMENT_TARGET_RADIUS,
            "minimal_height": OBJECT_LIFTED_HEIGHT,
        },
    )
    object_above_target = RewTerm(
        func=mdp.compute_object_above_target_reward,
        weight=0.0,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": PLACEMENT_TARGET_RADIUS,
            "minimal_height": OBJECT_LIFTED_HEIGHT,
        },
    )
    object_lowering_into_bowl = RewTerm(
        func=mdp.compute_object_lowering_into_bowl_reward,
        weight=10.0,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": BOWL_LOWERING_RADIUS,
            "target_height": BOWL_LOWERING_TARGET_HEIGHT,
            "height_std": 0.08,
            "minimal_height": BOWL_SUCCESS_MIN_HEIGHT,
        },
    )
    object_in_bowl = RewTerm(
        func=mdp.compute_object_in_bowl_success_reward,
        weight=1500.0,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": BOWL_SUCCESS_RADIUS,
            "min_height": BOWL_SUCCESS_MIN_HEIGHT,
            "max_height": BOWL_SUCCESS_MAX_HEIGHT,
            "max_speed": BOWL_SUCCESS_MAX_SPEED,
            "max_angular_speed": BOWL_SUCCESS_MAX_ANGULAR_SPEED,
            "min_gripper_open": BOWL_SUCCESS_MIN_GRIPPER_OPEN,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )
    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_velocity_penalty = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")}
    )


# Defines when an episode should stop.
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
    object_in_bowl = DoneTerm(
        func=mdp.terminate_on_object_in_bowl_success,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": BOWL_SUCCESS_RADIUS,
            "min_height": BOWL_SUCCESS_MIN_HEIGHT,
            "max_height": BOWL_SUCCESS_MAX_HEIGHT,
            "max_speed": BOWL_SUCCESS_MAX_SPEED,
            "max_angular_speed": BOWL_SUCCESS_MAX_ANGULAR_SPEED,
            "min_gripper_open": BOWL_SUCCESS_MIN_GRIPPER_OPEN,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )


# Keeps movement penalties fixed while the placement skill is still being learned.
@configclass
class CurriculumCfg:
    """Curriculum terms for the placement MDP."""

    pass


##
# Environment configuration
##


# Connects the scene, actions, observations, rewards, and stopping rules.
@configclass
class ObjectInBowlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ObjectInBowlSceneCfg = ObjectInBowlSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Post initialization
    # Fills in timing, camera, and simulator settings after the config is created.
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # viewer settings
        self.viewer.eye = (12.0, -12.0, 8.0)
        self.viewer.lookat = (0.0, 0.0, 0.1)
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
