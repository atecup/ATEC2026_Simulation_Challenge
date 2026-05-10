# Created by skywoodsz on 2026/02/06.
"""
ATEC environment base class, implementing basic I/O.
"""

import isaaclab.sim as sim_utils
from dataclasses import MISSING
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import mdp, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import atec_rl_lab.tasks.task_base.mdp as atec_mdp
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.assets.robots import ATECArticulationCfg
from .terrain_base import TerrainImporterCfg

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Base configuration for the scene."""
    terrain: TerrainImporterCfg = MISSING
    robot: ArticulationCfg = ATECArticulationCfg

    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ATEC_ASSETS_MODEL_DIR}/scene/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # Sensors
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    lidar_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.1,
        pattern_cfg=patterns.LidarPatternCfg(
            vertical_fov_range=(-20.0, 20.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
            channels=16,
        ),
        max_distance=10.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    head_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_camera",
        update_period = 0.1,
        height = 480,
        width = 640,
        data_types = ["rgb", "depth"],
        spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset = CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),  
        ),
    )

    ee_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ee_dual_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ee_dual_camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 50.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # For leg joint
    joint_pos_leg = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    )
    # For wheel joint
    joint_vel_wheel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )
    # For manipulator joint
    joint_pos_arm = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.5, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ProprioObservationsCfg(ObsGroup):
        """Observations for proprioception group."""
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ExteroObservationsCfg(ObsGroup):
        """Observations for exteroception group."""

        # observation terms (order preserved)
        lidar_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImageObservationsCfg(ObsGroup):
        """Observations for image group."""

        # observation terms (order preserved)
        head_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb", "normalize": False,},
        )
        head_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "depth"},
        )

        ee_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False,},
        )
        ee_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "depth"},
        )

        ee_dual_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_dual_camera"), "data_type": "rgb", "normalize": False,},
        )
        ee_dual_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_dual_camera"), "data_type": "depth"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    proprio: ProprioObservationsCfg = ProprioObservationsCfg()
    extero: ExteroObservationsCfg = ExteroObservationsCfg()
    image: ImageObservationsCfg = ImageObservationsCfg()

@configclass
class EventCfg:
    """Configuration for events."""
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base.*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # elapsed_time = RewTerm(func=atec_mdp.elapsed_time, weight=1)
    distance_from_origin = RewTerm(
        func=atec_mdp.distance_from_origin,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=1,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names="base.*"),
            "threshold": 1.0,
        },
    )
    fall = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": 0.0,
        },
        time_out=False,
    )
    # bad_orientation = DoneTerm(
    #     func=mdp.bad_orientation,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "limit_angle": 0.7
    #     },
    #     time_out=False,
    # )

class BaseEnvCfg(ManagerBasedRLEnvCfg):
    """Base environment configuration."""

    # Scene settings
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # MDP settings
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 1200 # 20 min
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # sensor feedback settings
        self.scene.contact_sensor.update_period = self.sim.dt * self.decimation

        robot_cfg = self.scene.robot
        base_path = "{ENV_REGEX_NS}/Robot"

        lidar_sensor_link_name = getattr(robot_cfg, "lidar_sensor_link_name", None)
        if isinstance(lidar_sensor_link_name, str):
            self.scene.lidar_sensor.prim_path = base_path + "/" + lidar_sensor_link_name
        else:
            self.scene.lidar_sensor = None
            self.observations.extero.lidar_scan = None

        head_camera_link_name = getattr(robot_cfg, "head_camera_link_name", None)
        head_camera_offset = getattr(robot_cfg, "head_camera_offset", None)
        if isinstance(head_camera_link_name, str):
            self.scene.head_camera.prim_path = base_path + "/" + head_camera_link_name + "/head_camera"
            if head_camera_offset is not None:
                self.scene.head_camera.offset = head_camera_offset
        else:
            self.scene.head_camera = None
            self.observations.image.head_rgb = None
            self.observations.image.head_depth = None

        ee_camera_link_name = getattr(robot_cfg, "ee_camera_link_name", None)
        ee_camera_offset = getattr(robot_cfg, "ee_camera_offset", None)
        if isinstance(ee_camera_link_name, str):
            self.scene.ee_camera.prim_path = base_path + "/" + ee_camera_link_name + "/ee_camera"
            if ee_camera_offset is not None:
                self.scene.ee_camera.offset = ee_camera_offset[0] if isinstance(ee_camera_offset, tuple) else ee_camera_offset
            self.scene.ee_dual_camera = None
            self.observations.image.ee_dual_rgb = None
            self.observations.image.ee_dual_depth = None
        elif isinstance(ee_camera_link_name, tuple):
            self.scene.ee_camera.prim_path = base_path + "/" + ee_camera_link_name[0] + "/ee_camera"
            self.scene.ee_dual_camera.prim_path = base_path + "/" + ee_camera_link_name[1] + "/ee_dual_camera"
            if ee_camera_offset is not None:
                if isinstance(ee_camera_offset, tuple):
                    # Allow independent offsets for left/right EE cameras.
                    self.scene.ee_camera.offset = ee_camera_offset[0]
                    self.scene.ee_dual_camera.offset = ee_camera_offset[1]
                else:
                    self.scene.ee_camera.offset = ee_camera_offset
                    self.scene.ee_dual_camera.offset = ee_camera_offset
        else:
            self.scene.ee_camera = None
            self.observations.image.ee_rgb = None
            self.observations.image.ee_depth = None
            
            self.scene.ee_dual_camera = None
            self.observations.image.ee_dual_rgb = None
            self.observations.image.ee_dual_depth = None

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = getattr(robot_cfg, "base_link_name", "base.*")
