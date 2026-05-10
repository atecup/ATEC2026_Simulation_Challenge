
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.assets.robots.cfg import ATECArticulationCfg
from isaaclab.sensors import CameraCfg
from scipy.spatial.transform import Rotation as R
import numpy as np

PIPER_USD_PATH = os.path.join(ATEC_ASSETS_MODEL_DIR, "robot/piper/piper.usd")

"""Configuration of Piper robot from Agilex.

This configuration loads the Piper robot USD file from the local asset directory
and sets up implicit actuators for all joints.
"""

PIPER_CFG = ATECArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD_PATH,
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
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        rot=(0.0, 0.0, 0.0, 1.0),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "default": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=800.0,
            damping=80.0,
        ),
    },
    ee_camera_link_name = "gripper_base",
    ee_camera_offset = CameraCfg.OffsetCfg(
        pos=(-0.05, 0.0, 0.06),
        rot=tuple(float(x) for x in R.from_euler("xyz", [0.0, 0.0, -np.pi / 2]).as_quat(scalar_first=True)),
        convention="ros",
    ),
    head_camera_link_name = None
)

PIPER_CFG.joint_names = [
    'joint1', 'joint2', 'joint3', 'joint4',
    'joint5', 'joint6', 'joint7', 'joint8'
]
