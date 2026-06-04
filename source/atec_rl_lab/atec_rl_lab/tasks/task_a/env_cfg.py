# Created by skywoodsz on 2026/02/06.

"""
Implementation of Task A environment configuration with different robots.
"""

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from atec_rl_lab.tasks.task_base import TerminationsCfg
from isaaclab.managers import SceneEntityCfg

from atec_rl_lab.tasks.task_base import BaseEnvCfg
from .terrain import TASK_A_TERRAIN_CFG
import atec_rl_lab.tasks.task_a.mdp as atec_mdp

@configclass
class TaskATerminationsCfg(TerminationsCfg):
    reach_goal_x = DoneTerm(
        func=atec_mdp.robot_x_greater_than,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_threshold": 145.0,
        },
        time_out=False,
    )

@configclass
class RewardsCfg:
    progress_reward = RewTerm(
        func=atec_mdp.CrossXMulti,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x_thresholds": [-115.0, -35.0, 45.0, 125.0, 140.0],
            "rewards": [2.0, 4.0, 8.0, 8.0, 4.0],
            "debug": False,
            "visual_assets": True,
        },
        weight=1.0,
    )


@configclass
class TaskAEnvCfg(BaseEnvCfg):
    """Environment base configuration for Task A."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TASK_A_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.rewards = RewardsCfg()
        self.terminations = TaskATerminationsCfg()

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.observations.image.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        # self.events.reset_robot_joints = None

        self.terminations.fall.params["minimum_height"] = -20.0



@configclass
class TaskAEnvG1Cfg(TaskAEnvCfg):
    """Environment configuration for Task A with Unitree g1."""
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG

        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=(-141, 0, 0.8),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_G1_29DOF_DEX1_CFG.base_link_name,
            ".*_hip_(pitch|roll|yaw)_link"
        ]

        joint_names = UNITREE_G1_29DOF_DEX1_CFG.joint_names
        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names
        self.actions.joint_leg.joint_names = joint_names
        self.actions.joint_wheel = None
        self.actions.joint_arm = None


@configclass
class TaskAEnvTron1Cfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON1A_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON1A_PIPER_CFG.base_link_name,
            "abad_[LR]_Link"
        ]

        joint_names = TRON1A_PIPER_CFG.joint_names
        leg_joint_names = TRON1A_PIPER_CFG.leg_joint_names
        wheel_joint_names = TRON1A_PIPER_CFG.wheel_joint_names
        arm_joint_names = TRON1A_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvTron2ALeggedCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_LEGGED_CFG

        self.scene.robot = TRON2A_LEGGED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_LEGGED_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON2A_LEGGED_CFG.base_link_name,
            "proximal_.*_[LR]_Link",
        ]

        joint_names = TRON2A_LEGGED_CFG.joint_names
        leg_joint_names = TRON2A_LEGGED_CFG.leg_joint_names
        arm_joint_names = TRON2A_LEGGED_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel = None
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvTron2AWheelCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_WHEEL_CFG

        self.scene.robot = TRON2A_WHEEL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_WHEEL_CFG.init_state.replace(
                pos=(-141, 0, 0.8 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON2A_WHEEL_CFG.base_link_name,
            "proximal_.*_[LR]_Link",
        ]

        joint_names = TRON2A_WHEEL_CFG.joint_names
        leg_joint_names = TRON2A_WHEEL_CFG.leg_joint_names
        wheel_joint_names = TRON2A_WHEEL_CFG.wheel_joint_names
        arm_joint_names = TRON2A_WHEEL_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names


@configclass
class TaskAEnvB2Cfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.58),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh"
        ]

        joint_names = UNITREE_B2_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2_PIPER_CFG.leg_joint_names
        arm_joint_names = UNITREE_B2_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names
        self.actions.joint_wheel = None

@configclass
class TaskAEnvB2WCfg(TaskAEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=(-141, 0, 0.78), 
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2W_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh"
        ]

        joint_names = UNITREE_B2W_PIPER_CFG.joint_names
        leg_joint_names = UNITREE_B2W_PIPER_CFG.leg_joint_names
        wheel_joint_names = UNITREE_B2W_PIPER_CFG.wheel_joint_names
        arm_joint_names = UNITREE_B2W_PIPER_CFG.arm_joint_names

        self.observations.proprio.joint_pos.params["asset_cfg"].joint_names = joint_names
        self.observations.proprio.joint_vel.params["asset_cfg"].joint_names = joint_names

        self.actions.joint_leg.joint_names = leg_joint_names
        self.actions.joint_wheel.joint_names = wheel_joint_names
        self.actions.joint_arm.joint_names = arm_joint_names
