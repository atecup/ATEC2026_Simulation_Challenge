
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg, CameraCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

"""
This file contains the configuration for the LIMX Tron1A robots.
"""

TRON1A_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/tron1/tron1a.usd")
TRON1A_PIPER_USD_PATH = os.path.join(
    ATEC_ASSETS_MODEL_DIR, "robot/tron1/tron1a_piper.usda"
)

TRON1A_WHEEL_CFG = ATECArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TRON1A_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
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
        activate_contact_sensors=True,
    ),
    init_state=ATECArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8 + 0.166),
        joint_pos={
            ".*_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit=80.0,
            velocity_limit=15.0,
            stiffness=40.0,
            damping=2.5,
            friction=0.01,
            armature=0.01,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "wheel_L_Joint",
                "wheel_R_Joint",
            ],
            effort_limit=80.0,
            velocity_limit=15.0,
            stiffness=0.0,
            damping=0.8,
            friction=0.01,
            armature=0.01,
        ),
    },
    base_link_name="base_Link",
    lidar_sensor_link_name="base_Link",
    head_camera_link_name="base_Link",
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.12, 0.0, 0.0),
        rot=tuple(float(x) for x in R.from_euler("xyz", [0., np.pi/6, 0.]).as_quat(scalar_first=True)),
        convention="world",
    ),
)

TRON1A_PIPER_CFG = deepcopy(TRON1A_WHEEL_CFG)
TRON1A_PIPER_CFG.spawn.usd_path = TRON1A_PIPER_USD_PATH
TRON1A_PIPER_CFG.init_state.joint_pos = {
    ".*_Joint": 0.0,
    "arm_joint[1-8]": 0.0,
}
TRON1A_PIPER_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr="arm_joint.*",
    effort_limit_sim=100.0,
    velocity_limit_sim=100.0,
    stiffness=80.0,
    damping=4.0,
    friction=0.01,
    armature=0.01,
)
TRON1A_PIPER_CFG.ee_camera_link_name = "gripper_base"
TRON1A_PIPER_CFG.ee_camera_offset = CameraCfg.OffsetCfg(
    pos=(-0.05, 0.0, 0.06),
    rot=tuple(float(x) for x in R.from_euler("xyz", [0., 0, -np.pi/2]).as_quat(scalar_first=True)),
    convention="ros",
)
TRON1A_PIPER_CFG.leg_joint_names = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",
]
TRON1A_PIPER_CFG.wheel_joint_names = [
    "wheel_L_Joint", "wheel_R_Joint",
]
TRON1A_PIPER_CFG.arm_joint_names = [
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]
TRON1A_PIPER_CFG.joint_names = [
    "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",
    "wheel_L_Joint", "wheel_R_Joint",
    'arm_joint1', 'arm_joint2', 'arm_joint3', 'arm_joint4',
    'arm_joint5', 'arm_joint6', 'arm_joint7', 'arm_joint8'
]