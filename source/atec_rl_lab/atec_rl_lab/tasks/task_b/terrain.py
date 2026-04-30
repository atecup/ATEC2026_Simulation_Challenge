from __future__ import annotations

from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
)
import trimesh
import numpy as np
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter

def flat_terrain_with_trash_bin(
    difficulty: float, cfg: FlatTerrainWithTrashBinCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    mesh_list = []

    # -------------------------
    # Ground: square plane centered at (0, 0)
    # -------------------------
    ground = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], 0.10),
        transform=trimesh.transformations.translation_matrix((0.0, 0.0, -0.005)),
    )
    mesh_list.append(ground)

    # -------------------------
    # Trash bin (relative to terrain center)
    # -------------------------
    bin_x = cfg.trash_bin_x
    bin_y = cfg.trash_bin_y if hasattr(cfg, "trash_bin_y") else 0.0

    bin_diameter = 2.0
    bin_radius = 0.5 * bin_diameter
    bin_height = 0.5
    wall_thickness = 0.02
    bottom_thickness = 0.05

    bin_color = np.asarray([255, 128, 0, 255], dtype=np.uint8)

    # Bottom: solid circular plate.
    bottom = trimesh.creation.cylinder(
        radius=bin_radius,
        height=bottom_thickness,
        transform=trimesh.transformations.translation_matrix((bin_x, bin_y, bottom_thickness / 2)),
    )
    bottom.visual.vertex_colors = np.tile(bin_color, (bottom.vertices.shape[0], 1))
    mesh_list.append(bottom)

    # Wall: annulus ring as circular bin wall.
    wall = trimesh.creation.annulus(
        r_min=max(0.0, bin_radius - wall_thickness),
        r_max=bin_radius,
        height=bin_height,
        transform=trimesh.transformations.translation_matrix((bin_x, bin_y, bottom_thickness + bin_height / 2)),
    )
    wall.visual.vertex_colors = np.tile(bin_color, (wall.vertices.shape[0], 1))
    mesh_list.append(wall)

    # -------------------------
    # Robot spawn origin: terrain center
    # -------------------------
    origin = np.array([0.0, 0.0, 0.0])

    return mesh_list, origin



@configclass
class FlatTerrainWithTrashBinCfg(SubTerrainBaseCfg):
    """Configuration for flat terrain with a trash bin at x=4.5.

    The trash bin is an orange hollow circle with open top.
    """
    function = flat_terrain_with_trash_bin
    trash_bin_x: float = 7  # Position relative to origin


TASK_B_TERRAIN_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(20, 20),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "flat_with_bin": FlatTerrainWithTrashBinCfg(proportion=1.0),
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
    # visual_material=None,
    debug_vis=False,
)
