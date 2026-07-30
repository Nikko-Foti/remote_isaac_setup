# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import observations, rewards

##
# Pre-defined configs
##

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort:skip
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort:skip


OBJECT_START_POSITION = (0.50, 0.0, 0.055)
# Height threshold for the binary lift reward.
OBJECT_LIFTED_HEIGHT = OBJECT_START_POSITION[2] + 0.05
LIFT_PROGRESS_REWARD_WEIGHT = 20.0
LIFT_PROGRESS_NEAR_OBJECT_DISTANCE = 0.08
VERIFIED_GRASP_FORCE_THRESHOLD = 1.0
VERIFIED_GRASP_HISTORY_LENGTH = 2
OBJECT_TO_BOWL_XY_REWARD_WEIGHT = 40.0
OBJECT_TO_BOWL_XY_REWARD_STD = 0.30
PLACEMENT_TARGET_XY = (0.70, 0.20)
PLACEMENT_TARGET_RADIUS = 0.08
TIGHT_TARGET_ENTRY_BONUS_WEIGHT = 1200.0
BOWL_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/024_bowl.usd"
BOWL_ASSET_POSITION = (PLACEMENT_TARGET_XY[0], PLACEMENT_TARGET_XY[1], 0.025)
BOWL_ASSET_ROTATION = (0.7071068, -0.7071068, 0.0, 0.0)
BOWL_SUCCESS_RADIUS = 0.11
BOWL_COLLISION_INNER_HALF_SIZE = BOWL_SUCCESS_RADIUS
BOWL_COLLISION_WALL_THICKNESS = 0.02
BOWL_COLLISION_WALL_HEIGHT = 0.08
BOWL_COLLISION_BASE_THICKNESS = 0.012
BOWL_COLLISION_OUTER_SIZE = 2.0 * (BOWL_COLLISION_INNER_HALF_SIZE + BOWL_COLLISION_WALL_THICKNESS)
BOWL_COLLISION_OBJECT_HALF_HEIGHT = 0.024
BOWL_COLLISION_FLOOR_TOP_Z = 0.025
PLACEMENT_TARGET_POSITION = (
    PLACEMENT_TARGET_XY[0],
    PLACEMENT_TARGET_XY[1],
    BOWL_COLLISION_FLOOR_TOP_Z + BOWL_COLLISION_OBJECT_HALF_HEIGHT,
)
BOWL_COLLISION_BASE_CENTER_Z = BOWL_COLLISION_FLOOR_TOP_Z - BOWL_COLLISION_BASE_THICKNESS / 2.0
BOWL_COLLISION_WALL_CENTER_Z = BOWL_COLLISION_FLOOR_TOP_Z + BOWL_COLLISION_WALL_HEIGHT / 2.0
BOWL_LOWERING_RADIUS = BOWL_SUCCESS_RADIUS
BOWL_LOWERING_REWARD_MIN_HEIGHT = OBJECT_START_POSITION[2] + 0.035
BOWL_SUCCESS_MIN_HEIGHT = PLACEMENT_TARGET_POSITION[2] - 0.005
BOWL_SUCCESS_MAX_HEIGHT = PLACEMENT_TARGET_POSITION[2] + 0.01
BOWL_LOWERING_TARGET_HEIGHT = PLACEMENT_TARGET_POSITION[2]
BOWL_SUCCESS_MAX_SPEED = 0.05
BOWL_SUCCESS_MAX_ANGULAR_SPEED = 0.5
BOWL_SUCCESS_MIN_GRIPPER_OPEN = 0.03
BOWL_SUPPORT_FORCE_THRESHOLD = 0.05
BOWL_RELEASE_CONTACT_FORCE_THRESHOLD = 0.05
BOWL_SUCCESS_DWELL_STEPS = 10
STRICT_PLACEMENT_SUCCESS_REWARD_WEIGHT = 2000.0
ARM_ACTION_RATE_PENALTY_WEIGHT = -0.5
PLACEMENT_SPEED_FREE_THRESHOLD = 0.10
PLACEMENT_SPEED_GATE_MAX_HEIGHT = 0.20


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
    robot.spawn.activate_contact_sensors = True

    left_finger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
        update_period=0.0,
        history_length=VERIFIED_GRASP_HISTORY_LENGTH,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    right_finger_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
        update_period=0.0,
        history_length=VERIFIED_GRASP_HISTORY_LENGTH,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

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
            activate_contact_sensors=True,
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

    object_bowl_support_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
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
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[PLACEMENT_TARGET_POSITION[0], PLACEMENT_TARGET_POSITION[1], BOWL_COLLISION_BASE_CENTER_Z]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_OUTER_SIZE, BOWL_COLLISION_BASE_THICKNESS),
            visible=False,
            collision_props=CollisionPropertiesCfg(),
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
            visible=False,
            collision_props=CollisionPropertiesCfg(),
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
            visible=False,
            collision_props=CollisionPropertiesCfg(),
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
            visible=False,
            collision_props=CollisionPropertiesCfg(),
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
            visible=False,
            collision_props=CollisionPropertiesCfg(),
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


# Gives the policy the fixed bowl target used by placement rewards.
@configclass
class CommandsCfg:
    """Command terms used by observations and later placement experiments."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="panda_hand",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(PLACEMENT_TARGET_POSITION[0], PLACEMENT_TARGET_POSITION[0]),
            pos_y=(PLACEMENT_TARGET_POSITION[1], PLACEMENT_TARGET_POSITION[1]),
            pos_z=(PLACEMENT_TARGET_POSITION[2], PLACEMENT_TARGET_POSITION[2]),
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
        object_position = ObsTerm(func=observations.get_object_position_in_robot_root_frame)
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
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.10, 0.10), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


# Defines the lift-only baseline rewards.
@configclass
class RewardsCfg:
    """Reward terms for proving the robot can reach and lift the cube."""

    # Logs diagnostics without changing the reward value.
    episode_diagnostics = RewTerm(
        func=rewards.update_episode_diagnostics,
        weight=1.0,
    )
    reaching_object = RewTerm(
        func=rewards.compute_reaching_object_reward,
        weight=1.0,
        params={"std": 0.10, "disable_after_lift_height": OBJECT_LIFTED_HEIGHT},
    )
    object_lift_progress = RewTerm(
        func=rewards.ComputeGatedObjectHeightProgressUntilTargetEntryReward,
        weight=LIFT_PROGRESS_REWARD_WEIGHT,
        params={
            "initial_height": OBJECT_START_POSITION[2],
            "target_height": OBJECT_LIFTED_HEIGHT,
            "near_distance": LIFT_PROGRESS_NEAR_OBJECT_DISTANCE,
            "target_position": PLACEMENT_TARGET_POSITION,
            "disable_radius": PLACEMENT_TARGET_RADIUS,
        },
    )
    tight_target_entry_bonus = RewTerm(
        func=rewards.ComputeFirstLiftedTargetEntryReward,
        weight=TIGHT_TARGET_ENTRY_BONUS_WEIGHT,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": PLACEMENT_TARGET_RADIUS,
            "minimal_height": OBJECT_LIFTED_HEIGHT,
        },
    )
    verified_grasp = RewTerm(
        func=rewards.compute_verified_grasp_reward,
        weight=2.0,
        params={
            "force_threshold": VERIFIED_GRASP_FORCE_THRESHOLD,
            "history_length": VERIFIED_GRASP_HISTORY_LENGTH,
            "initial_height": OBJECT_START_POSITION[2],
            "target_height": OBJECT_LIFTED_HEIGHT,
        },
    )
    object_to_bowl_xy = RewTerm(
        func=rewards.ComputeObjectToTargetXYProgressReward,
        weight=OBJECT_TO_BOWL_XY_REWARD_WEIGHT,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "std": OBJECT_TO_BOWL_XY_REWARD_STD,
            "radius": PLACEMENT_TARGET_RADIUS,
            "minimal_height": OBJECT_LIFTED_HEIGHT,
        },
    )
    strict_placement_success = RewTerm(
        func=rewards.compute_termination_reward,
        weight=STRICT_PLACEMENT_SUCCESS_REWARD_WEIGHT,
        params={"termination_name": "object_in_bowl"},
    )
    arm_action_rate_penalty = RewTerm(
        func=rewards.ComputeArmActionRatePenalty,
        weight=ARM_ACTION_RATE_PENALTY_WEIGHT,
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
        func=rewards.terminate_on_object_in_bowl_success,
        params={
            "target_position": PLACEMENT_TARGET_POSITION,
            "radius": BOWL_SUCCESS_RADIUS,
            "min_height": BOWL_SUCCESS_MIN_HEIGHT,
            "max_height": BOWL_SUCCESS_MAX_HEIGHT,
            "max_speed": BOWL_SUCCESS_MAX_SPEED,
            "max_angular_speed": BOWL_SUCCESS_MAX_ANGULAR_SPEED,
            "min_gripper_open": BOWL_SUCCESS_MIN_GRIPPER_OPEN,
            "support_force_threshold": BOWL_SUPPORT_FORCE_THRESHOLD,
            "finger_contact_force_threshold": BOWL_RELEASE_CONTACT_FORCE_THRESHOLD,
            "dwell_steps": BOWL_SUCCESS_DWELL_STEPS,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["panda_finger.*"]),
        },
    )


# Placeholder for future curriculum settings.
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

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
