# Created by skywoodsz on 2026/02/09.

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

import os
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_e/KLT_Bin/small_KLT.usd")

def Basket_cfg(pos, rot, name_suffix, scale=(1.5, 1.5, 0.5)):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=scale,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )
