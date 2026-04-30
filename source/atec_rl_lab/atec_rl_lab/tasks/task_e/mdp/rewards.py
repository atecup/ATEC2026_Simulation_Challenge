from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedEnv


class ObjectsInBasket(ManagerTermBase):
	"""Count Task E objects that enter the basket area for the first time.

	Reward is one-time per object per episode, so each object contributes at most +1.
	"""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._object_names = ("object_1", "object_2", "object_3")
		self._counted = torch.zeros(
			(self._env.num_envs, len(self._object_names)),
			device=self._env.device,
			dtype=torch.bool,
		)

	def reset(self, env_ids=None):
		if env_ids is None:
			self._counted.fill_(False)
		else:
			self._counted[env_ids] = False

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
			# Convert world coordinates to per-env local frame before region check.
			obj_pos = obj.data.root_pos_w[:, :3] - env.scene.env_origins
			inside_x = torch.abs(obj_pos[:, 0] - center_xy[0]) <= half_x
			inside_y = torch.abs(obj_pos[:, 1] - center_xy[1]) <= half_y
			inside_z = (obj_pos[:, 2] >= min_z) & (obj_pos[:, 2] <= max_z)
			inside_flags.append(inside_x & inside_y & inside_z)

		inside = torch.stack(inside_flags, dim=1)
		newly_inside = inside & (~self._counted)
		self._counted |= inside

		return newly_inside.sum(dim=1).to(torch.float32)


class GraspedObjectsByEE(ManagerTermBase):
	"""Give one-time reward per object when EE (gripper base) grasps it."""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._object_names = ("object_1", "object_2", "object_3")
		self._counted = torch.zeros(
			(self._env.num_envs, len(self._object_names)),
			device=self._env.device,
			dtype=torch.bool,
		)
		self._ee_body_idx = None

	def reset(self, env_ids=None):
		if env_ids is None:
			self._counted.fill_(False)
		else:
			self._counted[env_ids] = False

	def _ensure_ee_body_idx(self, env: ManagerBasedEnv, ee_body_name: str):
		if self._ee_body_idx is not None:
			return
		robot = env.scene["robot"]
		body_ids, _ = robot.find_bodies(ee_body_name)
		if len(body_ids) == 0:
			raise ValueError(
				f"Cannot find EE body by name regex '{ee_body_name}'."
			)
		self._ee_body_idx = int(body_ids[0])

	def __call__(
		self,
		env: ManagerBasedEnv,
		ee_body_name: str = "gripper_base",
		grasp_dist_thresh: float = 0.08,
		table_top_z: float = 0.8266,
		min_lift: float = 0.01,
		reward_per_object: float = 3.0,
	) -> torch.Tensor:
		self._ensure_ee_body_idx(env, ee_body_name)

		env_origins = env.scene.env_origins
		robot = env.scene["robot"]
		ee_pos = robot.data.body_pos_w[:, self._ee_body_idx, :3] - env_origins

		grasped_flags: list[torch.Tensor] = []
		for object_name in self._object_names:
			obj = env.scene[object_name]
			obj_pos = obj.data.root_pos_w[:, :3] - env_origins
			dist = torch.linalg.norm(ee_pos - obj_pos, dim=1)
			lifted = obj_pos[:, 2] >= (float(table_top_z) + float(min_lift))
			near = dist <= float(grasp_dist_thresh)
			grasped_flags.append(near & lifted)

		grasped = torch.stack(grasped_flags, dim=1)
		newly_grasped = grasped & (~self._counted)
		self._counted |= grasped

		return newly_grasped.sum(dim=1).to(torch.float32) * float(reward_per_object)
