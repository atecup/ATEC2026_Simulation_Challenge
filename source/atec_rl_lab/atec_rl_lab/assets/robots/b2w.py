"""
This file contains the configurations for the Unitree b2w and b2w with Piper arm mounted,
which are extended from the Unitree b2 robot configuration.
"""

import os
import numpy as np
from isaaclab.actuators import ImplicitActuatorCfg
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from isaaclab.sensors import CameraCfg
from atec_rl_lab.assets.robots import UNITREE_B2_CFG
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

B2W_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/b2w/b2w.usd")
B2W_PIPER_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/b2w/b2w_piper.usda")

UNITREE_B2W_CFG = deepcopy(UNITREE_B2_CFG)
UNITREE_B2W_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_B2W_CFG.spawn.usd_path = str(B2W_USD_PATH)
UNITREE_B2W_CFG.actuators["wheels"] = ImplicitActuatorCfg(
    joint_names_expr=".*_foot_joint",
    effort_limit_sim=20.0,
    velocity_limit_sim=50.0,
    stiffness=0.0,
    damping=1.0,
    friction=0.01,
    armature=0.01,
)

UNITREE_B2W_PIPER_CFG = deepcopy(UNITREE_B2W_CFG)
UNITREE_B2W_PIPER_CFG.spawn.articulation_props.enabled_self_collisions = False
UNITREE_B2W_PIPER_CFG.spawn.usd_path = str(B2W_PIPER_USD_PATH)
UNITREE_B2W_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr="arm_joint.*",
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)
UNITREE_B2W_PIPER_CFG.ee_camera_link_name = "gripper_base"
UNITREE_B2W_PIPER_CFG.ee_camera_offset = CameraCfg.OffsetCfg(
    pos=(-0.05, 0.0, 0.06),
    rot=tuple(float(x) for x in R.from_euler("xyz", [0., 0, -np.pi/2]).as_quat(scalar_first=True)),
    convention="ros",
)
UNITREE_B2W_PIPER_CFG.leg_joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
UNITREE_B2W_PIPER_CFG.wheel_joint_names = [
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
]
UNITREE_B2W_PIPER_CFG.arm_joint_names = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]
UNITREE_B2W_PIPER_CFG.joint_names = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]


