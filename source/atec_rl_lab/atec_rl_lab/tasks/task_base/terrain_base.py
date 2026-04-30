# Created by skywoodsz on 2026/02/06.
# copy from btx0424
from __future__ import annotations

import trimesh
import numpy as np
from isaaclab.terrains import (
    SubTerrainBaseCfg,
    TerrainImporterCfg,
    TerrainImporter,
    TerrainGenerator,
)
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.sim.spawners import materials

from atec_rl_lab.assets import ATEC_ASSETS_MODEL_DIR

class BetterTerrainImporter(TerrainImporter):
    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer. Different from IsaacLab's TerrainImporter,
        this class holds the terrain generator object.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        self.terrain_generator = None
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
            self.terrain_generator = terrain_generator
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def import_ground_plane(self, name: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            name: The name of the imported terrain. This name is used to create the USD prim
                corresponding to the terrain.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same name already exists.
        """
        # create prim path for the terrain
        prim_path = self.cfg.prim_path + f"/{name}"
        # check if key exists
        if prim_path in self.terrain_prim_paths:
            raise ValueError(
                f"A terrain with the name '{name}' already exists. Existing terrains: {', '.join(self.terrain_names)}."
            )
        # store the mesh name
        self.terrain_prim_paths.append(prim_path)

        # obtain ground plane color from the configured visual material
        color = (0.0, 0.0, 0.0)
        if self.cfg.visual_material is not None:
            material = self.cfg.visual_material.to_dict()
            # defaults to the `GroundPlaneCfg` color if diffuse color attribute is not found
            if "diffuse_color" in material:
                color = material["diffuse_color"]
            else:
                logger.warning(
                    "Visual material specified for ground plane but no diffuse color found."
                    " Using default color: (0.0, 0.0, 0.0)"
                )

        # get the mesh
        ground_plane_cfg = GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size, color=color)
        ground_plane_cfg.func(prim_path, ground_plane_cfg)

@configclass
class GroundPlaneCfg(SpawnerCfg):
    """Create a ground plane prim.

    This uses the USD for the standard grid-world ground plane from Isaac Sim by default.
    """

    func: Callable = spawn_ground_plane

    usd_path: str = f"{ATEC_ASSETS_MODEL_DIR}/scene/plane/default_environment.usd"
    """Path to the USD file to spawn asset from. Defaults to the grid-world ground plane."""

    color: tuple[float, float, float] | None = (0.0, 0.0, 0.0)
    """The color of the ground plane. Defaults to (0.0, 0.0, 0.0).

    If None, then the color remains unchanged.
    """

    size: tuple[float, float] = (100.0, 100.0)
    """The size of the ground plane. Defaults to 100 m x 100 m."""

    physics_material: materials.RigidBodyMaterialCfg = materials.RigidBodyMaterialCfg()
    """Physics material properties. Defaults to the default rigid body material."""

@configclass
class BetterTerrainGeneratorCfg(TerrainGeneratorCfg):
    terrain_sequence: list[str] | None = None

class BetterTerrainGenerator(TerrainGenerator):
    sub_terrain_types = []
    _cell_counter = 0

    def _get_terrain_mesh(
        self, difficulty: float, cfg: SubTerrainBaseCfg
    ) -> tuple[trimesh.Trimesh, np.ndarray]:

        seq = getattr(self.cfg, "terrain_sequence", None)
        if seq is not None:
            key = seq[self._cell_counter % len(seq)]
            if key not in self.cfg.sub_terrains:
                raise KeyError(
                    f"terrain_sequence key '{key}' not in cfg.sub_terrains: {list(self.cfg.sub_terrains.keys())}")
            cfg = self.cfg.sub_terrains[key]
        else:
            key = [k for k, v in self.cfg.sub_terrains.items() if v == cfg][0]

        self._cell_counter += 1
        self.sub_terrain_types.append(key)
        return super()._get_terrain_mesh(difficulty, cfg)