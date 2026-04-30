# Created by skywoodsz on 2026/02/06.

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter, BetterTerrainGeneratorCfg
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

TASK_A_TERRAIN_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=BetterTerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(20.0, 20.0), # for 300m
        border_width=0.0,
        num_rows=15,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
    terrain_sequence=[
            "flat", # -140
            "flat", # -120 start
            "random_rough", # -100
            "random_rough", # -80
            "random_rough", # -60
            "random_rough", # -40 base
            "hf_pyramid_slope", # -20
            "hf_pyramid_slope_inv", # 0
            "hf_pyramid_slope", # 20
            "hf_pyramid_slope_inv", # 40 slopes
            "pyramid_stairs", # 60
            "pyramid_stairs_inv", # 80
            "pyramid_stairs", # 100
            "pyramid_stairs_inv", # 120 # stairs
            "flat", # 140 # goal
        ],
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.1
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=(0.39, 0.40), platform_width=2.5, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.2, slope_range=(0.39, 0.40), platform_width=2.5, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.20),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2,
                step_height_range=(0.05, 0.20),
                step_width=0.3,
                platform_width=3.0,
                border_width=1.0,
                holes=False,
            ),
        },
    ),
    max_init_terrain_level=0,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=1.0,
    ),
    visual_material=sim_utils.MdlFileCfg(
        mdl_path=f"{ATEC_ASSETS_MODEL_DIR}/scene/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        project_uvw=True,
        texture_scale=(0.25, 0.25),
    ),
    debug_vis=False,
)
