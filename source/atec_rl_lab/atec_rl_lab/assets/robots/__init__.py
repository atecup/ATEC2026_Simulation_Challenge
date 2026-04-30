# Created by skywoodsz on 2026/01/28.

from .b2 import UNITREE_B2_CFG, UNITREE_B2_PIPER_CFG
from .b2w import UNITREE_B2W_CFG, UNITREE_B2W_PIPER_CFG
from .g1.g1_29dof_dex1 import UNITREE_G1_29DOF_DEX1_CFG
from atec_rl_lab.assets.robots.piper import PIPER_CFG
from .tron1a import TRON1A_WHEEL_CFG, TRON1A_PIPER_CFG
from .cfg import ATECArticulationCfg

ROBOTS = {
    "b2": UNITREE_B2_CFG,
    "b2w": UNITREE_B2W_CFG,
    "b2w_piper": UNITREE_B2W_PIPER_CFG,
    "b2_piper": UNITREE_B2_PIPER_CFG,
    "g1": UNITREE_G1_29DOF_DEX1_CFG,
    "piper": PIPER_CFG,
    "tron1a": TRON1A_WHEEL_CFG,
    "tron1a_piper": TRON1A_PIPER_CFG,
}

__all__ = [
    "UNITREE_B2_CFG",
    "UNITREE_B2W_CFG",
    "UNITREE_B2_PIPER_CFG",
    "UNITREE_B2W_PIPER_CFG",
    "UNITREE_G1_29DOF_DEX1_CFG",
    "PIPER_CFG",
    "TRON1A_WHEEL_CFG",
    "TRON1A_PIPER_CFG",
    "ROBOTS",
]
