# Created by skywoodsz on 2026/02/06.

"""
Implementation of Task D environment configuration with different robots.
"""

import copy
from isaaclab.utils import configclass
import atec_rl_lab.tasks.task_d.mdp as atec_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BaseEnvCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import TerminationsCfg as BaseTerminationsCfg
from .terrain import TASK_D_TERRAIN_CFG, PitAndPlatformTerrainCfg

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    achieve = RewTerm(
        func=atec_mdp.RewardCrossX,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "threshold": [-1.4, 2.0],
                "reward_value": [2, 20.0],
                "debug": False,
                "visual_assets": True,
                },
        weight=1.0,
    )
    box_in_target_x = RewTerm(
        func=atec_mdp.RewardBoxXInRange,
        params={
            "asset_cfg": SceneEntityCfg("box"),
            "x_min": [-0.7, -1.4],
            "x_max": [0.7, -0.7],
            "reward_value": 14.0,
            "one_time": True,
            "debug": False,
        },
        weight=1.0,
    )

@configclass
class TaskDTerminationsCfg(BaseTerminationsCfg):
    x_reached = DoneTerm(
        func=atec_mdp.robot_x_greater_than,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_threshold": 3.5,
        },
        time_out=False,
    )

@configclass
class TaskDEnvCfg(BaseEnvCfg):
    pit_width_range: tuple[float, float] = (1.3, 1.4)
    platform_height_range: tuple[float, float] = (1.0, 1.2)

    def _build_terrain_cfg(self):
        terrain_cfg = copy.deepcopy(TASK_D_TERRAIN_CFG)
        pit_cfg = terrain_cfg.terrain_generator.sub_terrains.get("pit_and_platform")
        if isinstance(pit_cfg, PitAndPlatformTerrainCfg):
            pit_cfg.pit_width_range = self.pit_width_range
            pit_cfg.platform_height_range = self.platform_height_range
        return terrain_cfg

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = self._build_terrain_cfg()
        self.scene.box = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Box",
            spawn=sim_utils.CuboidCfg(
                size=(0.8, 1.0, 0.6),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=8.0),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.9,
                    dynamic_friction=0.8,
                    restitution=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(-3, 1.6, 0.5), # -3, 1.6, 0.5
            ),
        )
        self.sim.physics_material = self.scene.terrain.physics_material

        # Task D reward
        self.rewards = RewardsCfg()
        self.terminations = TaskDTerminationsCfg()

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None

        # Trun off terminations
        self.terminations.illegal_contact = None
        self.terminations.fall.params["minimum_height"] = 0.25

@configclass
class TaskDEnvG1Cfg(TaskDEnvCfg):
    """Environment configuration for Task C with Unitree g1."""

    pit_width_range: tuple[float, float] = (0.9, 1.0)
    platform_height_range: tuple[float, float] = (0.9, 1.0)

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG

        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=(-3, 0.0, 0.8),
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_G1_29DOF_DEX1_CFG.base_link_name,
        #     ".*_hip_(pitch|roll|yaw)_link"
        # ]

        joint_names = UNITREE_G1_29DOF_DEX1_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos.joint_names = joint_names


@configclass
class TaskDEnvTron1Cfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = TRON1A_PIPER_CFG.init_state.replace(
                pos=(-3, 0.0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     TRON1A_PIPER_CFG.base_link_name,
        #     "abad_[LR]_Link",
        # ]

        joint_names = TRON1A_PIPER_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos.joint_names = joint_names


@configclass
class TaskDEnvB2Cfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(-3, 0.0, 0.8),
            )
        )

        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_B2_PIPER_CFG.base_link_name,
        #     ".*_hip",
        #     ".*_thigh",
        # ]

        joint_names = UNITREE_B2_PIPER_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos.joint_names = joint_names


@configclass
class TaskDEnvB2WCfg(TaskDEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state = UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=(-3, 0.0, 0.78),
            )
        )
        super().__post_init__()

        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
        #     UNITREE_B2W_PIPER_CFG.base_link_name,
        #     ".*_hip",
        #     ".*_thigh",
        # ]

        joint_names = UNITREE_B2W_PIPER_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_pos.joint_names = joint_names
