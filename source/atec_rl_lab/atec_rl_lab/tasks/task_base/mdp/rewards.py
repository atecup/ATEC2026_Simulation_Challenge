# Created by skywoodsz on 2026/02/07.

"""
ATEC scoring criteria
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING


from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def elapsed_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Return current elapsed episode time (seconds) for each env.

    Shape: (num_envs,)
    """
    t = env.common_step_counter * env.step_dt

    return torch.full( (env.num_envs,), float(t), device=env.device, dtype=torch.float32)


def distance_from_origin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Return distance from world origin for each env.

    Args:
        asset_cfg: Scene entity config for the asset whose root position is used.

    Shape: (num_envs,)
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    return torch.linalg.norm(root_pos_w, dim=-1)