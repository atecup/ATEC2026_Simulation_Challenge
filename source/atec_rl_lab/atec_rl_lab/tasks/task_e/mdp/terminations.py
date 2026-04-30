# Created by skywoodsz on 4/4/26.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedEnv


class ObjectsInBasketDone(ManagerTermBase):
	"""Terminate an episode once all Task-E objects are inside the basket region."""

	def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._object_names = ("object_1", "object_2", "object_3")

	def __call__(
		self,
		env: ManagerBasedEnv,
		center: tuple[float, float, float] = (1.08, -0.30, 0.74),
		half_x: float = 0.20,
		half_y: float = 0.11,
		table_top_z: float = 0.8266,
	) -> torch.Tensor:
		center_xy = torch.tensor(center[:2], device=env.device, dtype=torch.float32)
		env_origins_xy = env.scene.env_origins[:, :2]
		min_z = float(table_top_z)
		max_z = float(table_top_z) + 0.15

		inside_flags: list[torch.Tensor] = []
		for object_name in self._object_names:
			obj = env.scene[object_name]
			obj_pos = obj.data.root_pos_w[:, :3] - env.scene.env_origins
			inside_x = torch.abs(obj_pos[:, 0] - center_xy[0]) <= half_x
			inside_y = torch.abs(obj_pos[:, 1] - center_xy[1]) <= half_y
			inside_z = (obj_pos[:, 2] >= min_z) & (obj_pos[:, 2] <= max_z)
			inside_flags.append(inside_x & inside_y & inside_z)

		inside = torch.stack(inside_flags, dim=1)
		return inside.all(dim=1)
