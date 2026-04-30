
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

usd_path = os.path.join(ATEC_ASSETS_MODEL_DIR, "objects/task_e/shop_table/Shop_Table.usd")


def Table_cfg(pos, scale=(0.008, 0.008, 0.008)):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Shop_Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=scale
        )
    )
