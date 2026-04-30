from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils.configclass import configclass
from typing import Optional

EECameraOffsetCfg = CameraCfg.OffsetCfg | tuple[CameraCfg.OffsetCfg, CameraCfg.OffsetCfg]

@configclass
class ATECArticulationCfg(ArticulationCfg):
    base_link_name: str = "base_link"
    lidar_sensor_link_name: str = "base_link"
    head_camera_link_name: str = "base_link"
    ee_camera_link_name: Optional[str] = None # usually "gripper_base"
    head_camera_offset: CameraCfg.OffsetCfg = CameraCfg.OffsetCfg()
    ee_camera_offset: Optional[EECameraOffsetCfg] = None
    joint_names: list[str] = []
