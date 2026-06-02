# Created by skywoodsz on 5/15/26.


import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg, CameraCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

"""
This file contains the configuration for the LIMX Tron2A robots.
"""

TRON2A_SFYG_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/tron2/SFYG_TRON2A/robot.usd")
TRON2A_WFYG_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/tron2/WFYG_TRON2A/robot.usd")

TRON2A_LEGGED_CFG = ATECArticulationCfg(
spawn=sim_utils.UsdFileCfg(
        usd_path=str(TRON2A_SFYG_USD_PATH),
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
            # left leg
            "proximal_pitch_L_Joint": 0.0,
            "proximal_roll_L_Joint": 0.0,
            "proximal_yaw_L_Joint": -3.14159,
            "knee_L_Joint": 0.0,
            "ankle_pitch_L_Joint": 0.0,
            # right leg
            "proximal_pitch_R_Joint": 0.0,
            "proximal_roll_R_Joint": 0.0,
            "proximal_yaw_R_Joint": 3.14159,
            "knee_R_Joint": 0.0,
            "ankle_pitch_R_Joint": 0.0,
            # arm locked at zero
            "arm1_Joint": 0.0,
            "arm2_Joint": 0.0,
            "arm3_Joint": 0.0,
            "arm4_Joint": 0.0,
            "arm5_Joint": 0.0,
            "arm6_Joint": 0.0,
            "gripper1_Joint": 0.05,
            "gripper2_Joint": -0.05,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_large": ImplicitActuatorCfg(
            joint_names_expr=[
                "proximal_pitch_L_Joint",
                "proximal_roll_L_Joint",
                "knee_L_Joint",
                "proximal_pitch_R_Joint",
                "proximal_roll_R_Joint",
                "knee_R_Joint",
            ],
            effort_limit=150.0,
            velocity_limit=18.3,
            stiffness=159.67,
            damping=10.16,
            friction=0.01,
            armature=0.01,
        ),
        "legs_small": ImplicitActuatorCfg(
            joint_names_expr=[
                "proximal_yaw_L_Joint",
                "ankle_pitch_L_Joint",
                "proximal_yaw_R_Joint",
                "ankle_pitch_R_Joint"
            ],
            effort_limit=60.0,
            velocity_limit=22.52,
            stiffness=53.22,
            damping=3.39,
            friction=0.01,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm1_Joint", "arm2_Joint", "arm3_Joint",
                "arm4_Joint", "arm5_Joint", "arm6_Joint"
            ],
            effort_limit=100.0,
            velocity_limit=5.0,
            stiffness=80.0,
            damping=4.0,
            friction=0.01,
            armature=0.01,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "gripper1_Joint", "gripper2_Joint"
            ],
            effort_limit=10.0,
            velocity_limit=3.0,
            stiffness=80.0,
            damping=4.0,
            friction=0.01,
            armature=0.01,
        ),
    },
    base_link_name="base_Link",
    lidar_sensor_link_name="radar_Link",
    head_camera_link_name="base_imu",
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.22, 0.01, 0.18),
        rot=tuple(float(x) for x in R.from_euler(seq="xyz", angles=[0., 2.2689 - np.pi / 2, 0.]).as_quat(scalar_first=True)),
        convention="world",
    ),
)

TRON2A_LEGGED_CFG.ee_camera_link_name = "gripper_base_Link"
TRON2A_LEGGED_CFG.ee_camera_offset = CameraCfg.OffsetCfg(
    pos=(-0.05, 0.0, 0.06),
    rot=tuple(float(x) for x in R.from_euler(seq="xyz", angles=[0., 0, -np.pi/2]).as_quat(scalar_first=True)),
    convention="ros",
)
TRON2A_LEGGED_CFG.leg_joint_names = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "ankle_pitch_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
    "ankle_pitch_R_Joint"
]
TRON2A_LEGGED_CFG.arm_joint_names = [
    "arm1_Joint", "arm2_Joint", "arm3_Joint",
    "arm4_Joint", "arm5_Joint", "arm6_Joint",
    "gripper1_Joint", "gripper2_Joint"
]
TRON2A_LEGGED_CFG.joint_names = TRON2A_LEGGED_CFG.leg_joint_names + TRON2A_LEGGED_CFG.arm_joint_names

TRON2A_WHEEL_CFG = deepcopy(TRON2A_LEGGED_CFG)
TRON2A_WHEEL_CFG.spawn.usd_path = str(TRON2A_WFYG_USD_PATH)
TRON2A_WHEEL_CFG.init_state.joint_pos = {
    joint_name: joint_pos
    for joint_name, joint_pos in TRON2A_LEGGED_CFG.init_state.joint_pos.items()
    if joint_name not in (
        "proximal_yaw_L_Joint",
        "proximal_yaw_R_Joint",
        "ankle_pitch_L_Joint",
        "ankle_pitch_R_Joint",
    )
}
TRON2A_WHEEL_CFG.actuators["legs_small"].joint_names_expr = [
    "proximal_yaw_L_Joint",
    "proximal_yaw_R_Joint",
]
TRON2A_WHEEL_CFG.actuators["wheel"] = ImplicitActuatorCfg(
    joint_names_expr=[
        "wheel_L_Joint",
        "wheel_R_Joint",
    ],
    effort_limit=22.0,
    velocity_limit=41.89,
    stiffness=0.0,
    damping=0.8,
    friction=0.01,
    armature=0.01,
)
TRON2A_WHEEL_CFG.wheel_joint_names = [
    "wheel_L_Joint",
    "wheel_R_Joint",
]
TRON2A_WHEEL_CFG.leg_joint_names = [
    "proximal_pitch_L_Joint",
    "proximal_roll_L_Joint",
    "proximal_yaw_L_Joint",
    "knee_L_Joint",
    "proximal_pitch_R_Joint",
    "proximal_roll_R_Joint",
    "proximal_yaw_R_Joint",
    "knee_R_Joint",
]
TRON2A_WHEEL_CFG.joint_names = TRON2A_WHEEL_CFG.leg_joint_names + TRON2A_WHEEL_CFG.wheel_joint_names + TRON2A_WHEEL_CFG.arm_joint_names
