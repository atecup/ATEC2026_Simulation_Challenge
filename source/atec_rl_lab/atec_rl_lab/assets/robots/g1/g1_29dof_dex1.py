# Created by skywoodsz on 2026/03/15.

import os
from scipy.spatial.transform import Rotation as R
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg
import isaaclab.sim as sim_utils
from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

UNITREE_G1_29DOF_DEX1_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/g1/g1_29dof_with_dex1_rev_1_0.usd")

UNITREE_G1_29DOF_DEX1_CFG = ATECArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UNITREE_G1_29DOF_DEX1_USD_PATH}",
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
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
            # fingers joints
            "left_hand_Joint1_1": 0.024,
            "left_hand_Joint2_1": 0.024,
            "right_hand_Joint1_1": 0.024,
            "right_hand_Joint2_1": 0.024,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                ".*waist.*",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
                ".*waist_yaw_joint": 88.0,
                ".*waist_roll_joint": 35.0,
                ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
                ".*waist_yaw_joint": 32.0,
                ".*waist_roll_joint": 30.0,
                ".*waist_pitch_joint": 30.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                ".*waist.*": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                ".*waist.*": 5.0,
            },
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
            },
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
            },
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_wrist_.*",
            ],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_Joint1_1",
                "left_hand_Joint2_1",
                "right_hand_Joint1_1",
                "right_hand_Joint2_1",
            ],
            effort_limit=None,  # increase the torque limit
            velocity_limit=None,  # set the velocity limit to 0
            stiffness=800.0,  # increase the stiffness (kp)
            damping=3.0,  # increase the damping (kd)
            friction=200.0,
            armature=None,
        ),
    },
    base_link_name="torso_link",
    lidar_sensor_link_name="torso_link",
    head_camera_link_name="d435_link",
    head_camera_offset=CameraCfg.OffsetCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.5, -0.5, 0.5, -0.5),
        convention="ros",
    ),
    ee_camera_link_name=("left_hand_base_link", "right_hand_base_link"),
    ee_camera_offset=(
        CameraCfg.OffsetCfg(
            pos=(0.02541028, 0.045, 0.135),
            rot=(-0.34202, 0.93969, 0, 0),
            convention="ros",
        ),
        CameraCfg.OffsetCfg(
            pos=(-0.02541028, 0.045, 0.135),
            rot=(-0.34202, 0.93969, 0, 0),
            convention="ros",
        ),
    ),
    joint_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",

        "left_hand_Joint1_1",
        "left_hand_Joint2_1",
        "right_hand_Joint1_1",
        "right_hand_Joint2_1",
    ],
)
