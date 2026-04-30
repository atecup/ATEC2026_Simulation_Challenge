# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedEnv


class ObjectsInCircleDone(ManagerTermBase):
	"""Terminate when all 18 Task-B objects are inside the target circle."""

	def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._num_objects = 18

	def __call__(
		self,
		env: ManagerBasedEnv,
		center: tuple[float, float] = (-3.0, -10.0),
		radius: float = 1.0,
		z_min: float = 0.0,
		z_max: float = 0.5,
	) -> torch.Tensor:
		center_xy = torch.tensor(center, device=env.device, dtype=torch.float32)
		radius_sq = float(radius) * float(radius)

		inside_flags: list[torch.Tensor] = []
		for obj_idx in range(1, self._num_objects + 1):
			obj = env.scene[f"object_{obj_idx}"]
			obj_pos = obj.data.root_pos_w[:, :3]
			dist_sq = torch.sum((obj_pos[:, :2] - center_xy) ** 2, dim=1)
			inside_xy = dist_sq <= radius_sq
			inside_z = (obj_pos[:, 2] >= float(z_min)) & (obj_pos[:, 2] <= float(z_max))
			inside_flags.append(inside_xy & inside_z)

		inside = torch.stack(inside_flags, dim=1)
		return inside.all(dim=1)
