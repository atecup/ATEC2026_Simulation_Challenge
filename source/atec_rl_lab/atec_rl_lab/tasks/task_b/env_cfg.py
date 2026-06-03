# Created by skywoodsz on 2026/02/09.

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import AssetBaseCfg
import isaaclab.sim as sim_utils

from .terrain import TASK_B_TERRAIN_CFG
from atec_rl_lab.tasks.task_base import BaseEnvCfg, BaseSceneCfg
from atec_rl_lab.tasks.task_base.envs_base_cfg import TerminationsCfg as BaseTerminationsCfg
from atec_rl_lab.assets.objects import Sugar_cfg, Mustard_cfg, Banana_cfg, Cracker_cfg
from atec_rl_lab.tasks.task_b.mdp.terminations import ObjectsInCircleDone
import atec_rl_lab.tasks.task_b.mdp as atec_mdp

TARGET_CENTER = (-3.0, -10.0)
TARGET_MARKER_Z = 0.06

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    objects_in_circle = RewTerm(
        func=atec_mdp.ObjectsInCircle,
        params={"center": TARGET_CENTER,
                "radius": 1.0,
                "reward_per_object": 1.0},
        weight=1.0,
    )
    grasped_objects = RewTerm(
        func=atec_mdp.GraspedObjectsByEE,
        params={
            "ee_body_name": "gripper_base",
            "grasp_dist_thresh": 0.20,
            "reward_per_object": 1.0,
        },
        weight=1.0,
    )


@configclass
class TaskBTerminationsCfg(BaseTerminationsCfg):
    objects_in_circle_done = DoneTerm(
        func=ObjectsInCircleDone,
        params={"center": TARGET_CENTER, "radius": 1.0},
        time_out=False,
    )


@configclass
class TaskBEnvCfg(BaseEnvCfg):
    """Environment base configuration for Task C."""
    scene: BaseSceneCfg = BaseSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain = TASK_B_TERRAIN_CFG
        self.sim.physics_material = self.scene.terrain.physics_material
        self.rewards = RewardsCfg()
        self.terminations = TaskBTerminationsCfg()

        # Turn off the DR and noise
        self.observations.proprio.enable_corruption = False
        self.observations.extero.enable_corruption = False
        self.observations.image.enable_corruption = False
        self.events.physics_material = None
        self.events.base_external_force_torque = None
        # self.events.reset_robot_joints = None

        import numpy as np
        rng = np.random.default_rng(seed=self.seed)

        SUGAR_QUAT  = [0.0, 0.707, 0.0, 0.707]     
        OTHER_QUAT  = [0.0, 0.0, -0.707, 0.707]     

        for i in range(18):
            x = rng.uniform(-15.0, -5.0)
            y = rng.uniform(-15.0, -5.0)

            if abs(x) < 1.0 and abs(y) < 1.0:
                x += 2.0

            name = f"Object{i+1}"

            if i < 6:
                cfg = Sugar_cfg([x, y, 0.15], SUGAR_QUAT, name)
            elif i < 12:
                cfg = Mustard_cfg([x, y, 0.10], OTHER_QUAT, name)
            else:
                cfg = Banana_cfg([x, y, 0.10], OTHER_QUAT, name)

            setattr(self.scene, f"object_{i+1}", cfg)


@configclass
class TaskBEnvG1Cfg(TaskBEnvCfg):
    """Environment configuration for Task C with Unitree g1."""

    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_G1_29DOF_DEX1_CFG
        self.scene.robot = UNITREE_G1_29DOF_DEX1_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_G1_29DOF_DEX1_CFG.init_state.replace(
                pos=(-10, -10, 0.9),
            )
        )
        super().__post_init__()

        self.rewards.grasped_objects.params["ee_body_name"] = (
            "left_hand_base_link",
            "right_hand_base_link",
        )
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
class TaskBEnvTron1Cfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON1A_PIPER_CFG

        self.scene.robot = TRON1A_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON1A_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.9 + 0.166),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            TRON1A_PIPER_CFG.base_link_name,
            "abad_[LR]_Link",
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
class TaskBEnvTron2ALeggedCfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_LEGGED_CFG

        self.scene.robot = TRON2A_LEGGED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_LEGGED_CFG.init_state.replace(
                pos=(-10, -10, 0.9 + 0.166),
            )
        )
        super().__post_init__()

        self.rewards.grasped_objects.params["ee_body_name"] = "gripper_base_Link"
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
class TaskBEnvTron2AWheelCfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import TRON2A_WHEEL_CFG

        self.scene.robot = TRON2A_WHEEL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=TRON2A_WHEEL_CFG.init_state.replace(
                pos=(-10, -10, 0.9 + 0.166),
            )
        )
        super().__post_init__()

        self.rewards.grasped_objects.params["ee_body_name"] = "gripper_base_Link"
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
class TaskBEnvB2Cfg(TaskBEnvCfg):
    def __post_init__(self):

        from atec_rl_lab.assets.robots import UNITREE_B2_PIPER_CFG

        self.scene.robot = UNITREE_B2_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.68),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh",
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
class TaskBEnvB2WCfg(TaskBEnvCfg):
    def __post_init__(self):
        from atec_rl_lab.assets.robots import UNITREE_B2W_PIPER_CFG

        self.scene.robot = UNITREE_B2W_PIPER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=UNITREE_B2W_PIPER_CFG.init_state.replace(
                pos=(-10, -10, 0.78),
            )
        )
        super().__post_init__()

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            UNITREE_B2W_PIPER_CFG.base_link_name,
            ".*_hip",
            ".*_thigh",
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
