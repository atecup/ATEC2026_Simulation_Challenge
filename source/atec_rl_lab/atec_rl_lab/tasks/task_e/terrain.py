from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter, BetterTerrainGeneratorCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

TASK_E_TERRAIN_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,
)