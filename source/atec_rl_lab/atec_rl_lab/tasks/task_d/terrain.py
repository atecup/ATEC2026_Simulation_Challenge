from __future__ import annotations

import trimesh
import numpy as np
import random
import isaaclab.sim as sim_utils
from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainGeneratorCfg,
    TerrainImporterCfg,
    MeshPlaneTerrainCfg
)
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR
from atec_rl_lab.tasks.task_base import BetterTerrainGenerator, BetterTerrainImporter


def pit_and_platform_terrain(
    difficulty: float, cfg: PitAndPlatformTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain with a pit and an adjacent platform.

    The terrain contains a pit of fixed depth (0.4 m) with a bottom surface, and a platform
    positioned randomly on either the left or right side of the pit. The pit width and platform
    height are scaled linearly with the difficulty parameter based on their respective ranges
    in the configuration. The platform width matches the pit width, creating a choice for the
    robot to either traverse the pit or use the elevated platform to bypass it.

    Args:
        difficulty: Difficulty parameter in [0, 1] that scales the pit width and platform height
            between their configured minimum and maximum values.
        cfg: Configuration object containing terrain parameters such as size, pit width range,
            platform height range, and border width.

    Returns:
        A tuple containing:
            - A list of trimesh.Trimesh objects representing the terrain meshes (pit borders,
              pit bottom, and platform).
            - A numpy array of shape (3,) representing the origin position where the robot
              should be spawned, located at 25% along the x-axis and centered on the y-axis.
    """
    mesh_list = []
    pit_depth = cfg.pit_depth
    pit_width = cfg.pit_width_range[0] + difficulty * (
        cfg.pit_width_range[1] - cfg.pit_width_range[0]
    )
    platform_width = pit_width
    platform_height = cfg.platform_height_range[0] + difficulty * (
        cfg.platform_height_range[1] - cfg.platform_height_range[0]
    )
    mesh_list.extend(
        make_border(
            size=(cfg.size[0], cfg.size[1] - cfg.border_width),
            inner_size=(pit_width, cfg.size[1] - cfg.border_width - 0.2),
            height=pit_depth,
            position=(cfg.size[0] / 2, cfg.size[1] / 2, -pit_depth / 2),
        )
    )
    pit_bottom_thickness = 0.2
    pit_bottom = trimesh.creation.box(
        extents=(pit_width, cfg.size[1] - cfg.border_width, pit_bottom_thickness),
        transform=trimesh.transformations.translation_matrix(
            (cfg.size[0] / 2, cfg.size[1] / 2, -pit_depth - pit_bottom_thickness / 2)
        ),
    )
    # left_or_right = random.choice([0.25, 0.75])
    left_or_right = 0.75
    platform = trimesh.creation.box(
        extents=(platform_width, cfg.size[1] / 2 - cfg.border_width, platform_height),
        transform=trimesh.transformations.translation_matrix(
            (cfg.size[0] / 2, cfg.size[1] * left_or_right, platform_height / 2)
        ),
    )
    mesh_list.append(pit_bottom)
    mesh_list.append(platform)
    origin = np.array([cfg.size[0] * 0.15, cfg.size[1] / 2, 0.0])
    return mesh_list, origin


@configclass
class PitAndPlatformTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a terrain with a pit and an adjacent platform.

    This terrain type creates a navigation challenge where a robot must choose between
    traversing a pit or using an elevated platform positioned randomly on either the left
    or right side of the pit. The pit width and platform height are scaled by the difficulty
    parameter within their specified ranges.

    Attributes:
        border_width: Width of the border around the terrain (in m). Defaults to 0.5.
        pit_width_range: Minimum and maximum width of the pit (in m). The actual width
            is interpolated based on difficulty. Defaults to (0.8, 1.2).
        platform_height_range: Minimum and maximum height of the platform above ground
            (in m). The actual height is interpolated based on difficulty. Defaults to (0.3, 0.6).
    """

    function = pit_and_platform_terrain
    border_width: float = 1.0
    pit_depth: float = 1.0
    # wo random
    pit_width_range: tuple[float, float] = (1.6, 1.7)
    platform_height_range: tuple[float, float] = (1.4, 1.5)


def platform_terrain(difficulty: float, cfg: PlatformTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    platform_width = cfg.platform_width_range[0] + random.random() * (
        cfg.platform_width_range[1] - cfg.platform_width_range[0]
    )
    platform_height = cfg.platform_height_range[0] + difficulty * (
        cfg.platform_height_range[1] - cfg.platform_height_range[0]
    )
    mesh_list = []
    
    ground = trimesh.creation.box(
        extents=(cfg.size[0], cfg.size[1], 0.1),
        transform=trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, -0.05))
    )
    mesh_list.append(ground)

    platform = trimesh.creation.box(
        extents=(platform_width, cfg.size[1], platform_height),
        transform=trimesh.transformations.translation_matrix((cfg.size[0] / 2, cfg.size[1] / 2, platform_height / 2))
    )
    
    mesh_list.append(platform)
    origin = np.array([cfg.size[0] * 0.15, cfg.size[1] / 2, platform_height])
    return mesh_list, origin


@configclass
class PlatformTerrainCfg(SubTerrainBaseCfg):
    function = platform_terrain
    platform_width_range: tuple[float, float] = (1.9, 2.0)
    platform_height_range: tuple[float, float] = (0.5, 0.6)


TASK_D_TERRAIN_CFG = TerrainImporterCfg(
    class_type=BetterTerrainImporter,
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        class_type=BetterTerrainGenerator,
        seed=0,
        size=(12.0, 8.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains={
            "pit_and_platform": PitAndPlatformTerrainCfg(proportion=1.0),
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
