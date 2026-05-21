from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs import mdp
from isaaclab.sensors import CameraCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BaseEnvCfg, BaseSceneCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import ObservationsCfg as BaseObservationsCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import TerminationsCfg as BaseTerminationsCfg
from atec_rl_lab.assets.objects import Sugar_cfg, Mustard_cfg, Banana_cfg, Table_cfg, Basket_cfg
from atec_rl_lab.tasks.task_e.mdp.terminations import ObjectsInBasketDone
import atec_rl_lab.tasks.task_e.mdp as atec_mdp
from .terrain import TASK_E_TERRAIN_CFG

TABLE_CENTER_X = 1.00
TABLE_CENTER_Y = 0.00
TABLE_CENTER_Z = 0.00
TABLE_SCALE = 0.01
TABLE_DIMS_AT_0P008 = (0.6468062441005529, 0.9084968693231588, 0.6613141183247961)
TABLE_DIMS = tuple(dim * (TABLE_SCALE / 0.008) for dim in TABLE_DIMS_AT_0P008)
TABLE_HALF_X = TABLE_DIMS[0] * 0.5
TABLE_HALF_Y = TABLE_DIMS[1] * 0.5
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_DIMS[2]
# Per-object 2-D bounding-box half-extents (world XY, metres, scale=1).
# Must stay in sync with OBJ_HALF_EXTENTS in scripts/act/task_e/config.py.
OBJ_HALF_EXTENTS = {
    "object_1": (0.050, 0.044),   # Sugar box
    "object_2": (0.050, 0.030),   # Mustard bottle
    "object_3": (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015

BASKET_CENTER_X = TABLE_CENTER_X + 0.08
BASKET_CENTER_Y = TABLE_CENTER_Y - 0.30
BASKET_BASE_Z = TABLE_TOP_Z + TABLE_DIMS[2]
BASKET_EXCL_HALF_X = 0.29
BASKET_EXCL_HALF_Y = 0.32

BASKET_SUCCESS_CENTER = (BASKET_CENTER_X, BASKET_CENTER_Y, TABLE_TOP_Z + 0.15)
BASKET_SUCCESS_HALF_X = 0.20
BASKET_SUCCESS_HALF_Y = 0.11

# Task-E global camera used by ACT evaluation/deployment.
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)

@configclass
class TaskERewardsCfg:
    objects_in_basket = RewTerm(
        func=atec_mdp.ObjectsInBasket,
        params={
            "center": BASKET_SUCCESS_CENTER,
            "half_x": BASKET_SUCCESS_HALF_X,
            "half_y": BASKET_SUCCESS_HALF_Y,
            "table_top_z": TABLE_TOP_Z,
        },
        weight=3.0,
    )
    grasped_objects_once = RewTerm(
        func=atec_mdp.GraspedObjectsByEE,
        params={
            "ee_body_name": "gripper_base",
            "grasp_dist_thresh": 0.20,
            "table_top_z": TABLE_TOP_Z,
            "min_lift": 0.15,
            "reward_per_object": 3.0,
        },
        weight=1.0,
    )

@configclass
class TaskETerminationsCfg(BaseTerminationsCfg):
    basket_success = DoneTerm(
        func=ObjectsInBasketDone,
        params={
            "center": BASKET_SUCCESS_CENTER,
            "half_x": BASKET_SUCCESS_HALF_X,
            "half_y": BASKET_SUCCESS_HALF_Y,
            "table_top_z": TABLE_TOP_Z,
        },
        time_out=False,
    )

@configclass
class TaskEObservationsCfg(BaseObservationsCfg):
    """Task-E specific observations overriding BaseEnvCfg defaults."""

    @configclass
    class ProprioObservationsCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class ImageObservationsCfg(ObsGroup):
        ee_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "rgb", "normalize": False,},
        )
        ee_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("ee_camera"), "data_type": "depth"},
        )
        video_rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("video_cam"), "data_type": "rgb", "normalize": False,},
        )
        video_depth = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("video_cam"), "data_type": "depth"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    proprio: ProprioObservationsCfg = ProprioObservationsCfg()
    image: ImageObservationsCfg = ImageObservationsCfg()

@configclass
class TaskESceneCfg(BaseSceneCfg):
    """Scene configuration for tabletop pick-and-place with 3 objects."""

    table = Table_cfg([TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_CENTER_Z], scale=(TABLE_SCALE, TABLE_SCALE, TABLE_SCALE))

    object_1 = Sugar_cfg([TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z + 0.03], [0.0, 0.707, 0.0, 0.707], "Object1")
    object_2 = Mustard_cfg([TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z + 0.03], [0.0, 0.0, -0.707, 0.707], "Object2")
    object_3 = Banana_cfg([TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z + 0.03], [0.0, 0.0, -0.707, 0.707], "Object3")

    basket = Basket_cfg(
        [BASKET_CENTER_X, BASKET_CENTER_Y, TABLE_TOP_Z + 0.08],
        [0.707, 0, 0, 0.707],
        "Basket",
        scale=(1.6, 1.6, 1.0),
    )

    video_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/video_cam",
        update_period=0.0,
        height=CAM_H,
        width=CAM_W,
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
    )

class TaskEEnvCfg(BaseEnvCfg):
    """Environment configuration for Task E tabletop manipulation."""

    scene: TaskESceneCfg = TaskESceneCfg(num_envs=512, env_spacing=2.5)
    observations: TaskEObservationsCfg = TaskEObservationsCfg()
    rewards: TaskERewardsCfg = TaskERewardsCfg()
    terminations: TaskETerminationsCfg = TaskETerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TASK_E_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material

        # self.observations.proprio.enable_corruption = False
        self.observations.extero = None
        # self.observations.image = None
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        self.events.reset_robot_joints = None

        # Per-object Y-bands (must match OBJ_SPAWN_Y_BANDS in scripts/act/task_e/config.py)
        x_min, x_max = TABLE_CENTER_X - 0.10, TABLE_CENTER_X + 0.10
        z = TABLE_TOP_Z + 0.05
        y_bands = {
            "object_1": (TABLE_CENTER_Y + 0.25, TABLE_CENTER_Y + 0.29),
            "object_2": (TABLE_CENTER_Y + 0.14, TABLE_CENTER_Y + 0.20),
            "object_3": (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.09),
        }
        import numpy as np
        rng = np.random.default_rng(seed=self.seed)
        placed: dict[str, tuple[float, float]] = {}
        for object_name, (y_min, y_max) in y_bands.items():
            hx, hy = OBJ_HALF_EXTENTS[object_name]
            x = y = None
            for _ in range(200):
                cx = float(rng.uniform(x_min, x_max))
                cy = float(rng.uniform(y_min, y_max))
                ok = all(
                    abs(cx - px) >= hx + OBJ_HALF_EXTENTS[pn][0] + OBJ_BBOX_MARGIN or
                    abs(cy - py) >= hy + OBJ_HALF_EXTENTS[pn][1] + OBJ_BBOX_MARGIN
                    for pn, (px, py) in placed.items()
                )
                if ok:
                    x, y = cx, cy
                    break
            if x is None:  # fallback: band centre
                x = (x_min + x_max) / 2.0
                y = (y_min + y_max) / 2.0
            placed[object_name] = (x, y)
            getattr(self.scene, object_name).init_state.pos = (x, y, z)

@configclass
class TaskEEnvPiperCfg(TaskEEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import PIPER_CFG

        piper_cfg = PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=PIPER_CFG.init_state.replace(
                pos=(TABLE_CENTER_X + TABLE_HALF_X, TABLE_CENTER_Y, TABLE_TOP_Z),
            ),
        )
        piper_cfg.spawn.rigid_props.disable_gravity = True
        piper_cfg = piper_cfg.replace(
            init_state=piper_cfg.init_state.replace(
                joint_pos={
                    "joint1":  0.0,
                    "joint2":  1.2,   # pre-lifts arm above table
                    "joint3": -1.5,
                    "joint4":  0.0,
                    "joint5":  1.2,
                    "joint6":  0.0,
                    "joint7":  0.035, # gripper open
                    "joint8": -0.035,
                },
            )
        )
        self.scene.robot = piper_cfg
        super().__post_init__()

        self.commands.base_velocity = None
        self.terminations.base_contact = None

        self.observations.proprio.velocity_commands = None

        joint_names = PIPER_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_leg = None
        self.actions.joint_wheel = None
        self.actions.joint_arm.joint_names = joint_names
