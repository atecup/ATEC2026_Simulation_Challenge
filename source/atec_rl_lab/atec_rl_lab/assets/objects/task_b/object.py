# Created by skywoodsz on 2026/02/09.

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

cracker_usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_b/003_cracker_box.usd")
sugar_usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_b/004_sugar_box.usd")
mustard_usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_b/006_mustard_bottle.usd")
banana_usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_b/011_banana.usd")

def Banana_cfg(pos, rot, name_suffix):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=banana_usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )

def Mustard_cfg(pos, rot, name_suffix):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=mustard_usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )

def Sugar_cfg(pos, rot, name_suffix):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=sugar_usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )

def Cracker_cfg(pos, rot, name_suffix):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name_suffix}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cracker_usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                linear_damping=2.0,
                angular_damping=4.0,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.5,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.01,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
        ),
    )