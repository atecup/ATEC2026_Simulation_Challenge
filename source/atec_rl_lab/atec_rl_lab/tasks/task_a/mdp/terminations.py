from __future__ import annotations

from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_x_greater_than(
    env: ManagerBasedEnv,
    asset_cfg,
    x_threshold: float,
) -> torch.Tensor:
    """Terminate when robot world x is greater than x_threshold."""
    robot = env.scene[asset_cfg.name]
    robot_x = robot.data.root_pos_w[:, 0]   # world x

    return robot_x > x_threshold