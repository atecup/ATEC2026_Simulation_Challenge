# Created by skywoodsz on 2026/01/28.

"""
Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_rl_lab
"""

import os
import numpy as np
from copy import deepcopy
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg
from isaaclab.sensors import CameraCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from scipy.spatial.transform import Rotation as R

B2_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/b2/b2.usd")
B2_PIPER_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/b2/b2_piper.usda")

UNITREE_B2_CFG = ATECArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{B2_USD_PATH}",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ATECArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*R_hip_joint": -0.1,
            ".*L_hip_joint": 0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "M107-24-2": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_.*", ".*_thigh_.*"],
            effort_limit_sim=200,
            velocity_limit_sim=23,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
        "2": ImplicitActuatorCfg(
            joint_names_expr=[".*_calf_.*"],
            effort_limit_sim=320,
            velocity_limit_sim=14,
            stiffness=160.0,
            damping=5.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.4216099977493286, 0.02500000037252903, 0.06185099855065346),
        rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, np.pi / 6, 0.0]).as_quat(scalar_first=True)),
        convention="world",
    ),
)

UNITREE_B2_PIPER_CFG = deepcopy(UNITREE_B2_CFG)
UNITREE_B2_PIPER_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_B2_PIPER_CFG.spawn.usd_path = str(B2_PIPER_USD_PATH)
UNITREE_B2_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr="arm_joint.*",
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)
UNITREE_B2_PIPER_CFG.ee_camera_link_name = "gripper_base"
UNITREE_B2_PIPER_CFG.ee_camera_offset = CameraCfg.OffsetCfg(
    pos=(-0.05, 0.0, 0.06),
    rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, 0.0, -np.pi / 2]).as_quat(scalar_first=True)),
    convention="ros",
)

UNITREE_B2_PIPER_CFG.leg_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
UNITREE_B2_PIPER_CFG.arm_joint_names = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]
UNITREE_B2_PIPER_CFG.joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]
