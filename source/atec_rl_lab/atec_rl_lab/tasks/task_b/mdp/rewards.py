from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import RewardTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
	from isaaclab.envs import ManagerBasedEnv


class ObjectsInCircle(ManagerTermBase):
	"""Count Sugar/Mustard/Banana objects that enter a target circle for the first time.

	The reward is one-time per object per episode:
	once an object has been counted, it will not be rewarded again until reset.
	"""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._num_objects = 18
		self._counted = torch.zeros(
			(self._env.num_envs, self._num_objects),
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
		center: tuple[float, float] = (-3.0, -10.0),
		radius: float = 1.0,
		reward_per_object: float = 10.0,
		z_min: float = 0.0,
		z_max: float = 0.5,
	) -> torch.Tensor:
		center_xy = torch.tensor(center, device=env.device, dtype=torch.float32)
		radius_sq = radius * radius

		inside_flags: list[torch.Tensor] = []
		for obj_idx in range(1, self._num_objects + 1):
			obj = env.scene[f"object_{obj_idx}"]
			obj_pos = obj.data.root_pos_w[:, :3]
			dist_sq = torch.sum((obj_pos[:, :2] - center_xy) ** 2, dim=1)
			inside_xy = dist_sq <= radius_sq
			inside_z = (obj_pos[:, 2] >= float(z_min)) & (obj_pos[:, 2] <= float(z_max))
			inside_flags.append(inside_xy & inside_z)

		inside = torch.stack(inside_flags, dim=1)
		newly_inside = inside & (~self._counted)
		self._counted |= inside

		return newly_inside.sum(dim=1).to(torch.float32) * float(reward_per_object)


class GraspedObjectsByEE(ManagerTermBase):
	"""Count objects reached by one or two end-effectors for the first time.

	Each object contributes at most +1 per episode by default.
	"""

	def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
		super().__init__(cfg, env)
		self._num_objects = 18
		self._counted = torch.zeros(
			(self._env.num_envs, self._num_objects),
			device=self._env.device,
			dtype=torch.bool,
		)
		self._ee_body_idxs: list[int] | None = None
		self._ee_body_names_key: tuple[str, ...] | None = None

	def _normalize_ee_body_names(self, ee_body_name: str | tuple[str, str]) -> tuple[str, ...]:
		if isinstance(ee_body_name, str):
			return (ee_body_name,)
		if isinstance(ee_body_name, tuple):
			if len(ee_body_name) not in (1, 2):
				raise ValueError("ee_body_name tuple must contain one or two body names.")
			return tuple(str(name) for name in ee_body_name)
		raise TypeError(f"Unsupported ee_body_name type: {type(ee_body_name)}")

	def _ensure_ee_body_idxs(self, env: ManagerBasedEnv, ee_body_name: str | tuple[str, str]):
		names_key = self._normalize_ee_body_names(ee_body_name)
		if self._ee_body_idxs is not None and self._ee_body_names_key == names_key:
			return

		robot = env.scene["robot"]
		body_idxs: list[int] = []
		used_names: list[str] = []
		for name in names_key:
			body_ids, found_names = robot.find_bodies(name)
			if len(body_ids) == 0:
				raise ValueError(f"Cannot find EE body by name regex '{name}'.")
			body_idxs.append(int(body_ids[0]))
			used_names.append(found_names[0])

		self._ee_body_idxs = body_idxs
		self._ee_body_names_key = names_key
		print(f"[GraspedObjectsByEE] using ee bodies: {used_names} (idxs={body_idxs})")

	def reset(self, env_ids=None):
		if env_ids is None:
			self._counted.fill_(False)
		else:
			self._counted[env_ids] = False

	def __call__(
		self,
		env: ManagerBasedEnv,
		ee_body_name: str | tuple[str, str] = "gripper_base",
		grasp_dist_thresh: float = 0.12,
		reward_per_object: float = 1.0,
		distance_threshold: float | None = None,
	) -> torch.Tensor:
		# Backward compatibility: old cfg used distance_threshold.
		if distance_threshold is not None:
			grasp_dist_thresh = float(distance_threshold)

		self._ensure_ee_body_idxs(env, ee_body_name)

		robot = env.scene["robot"]
		ee_pos = robot.data.body_pos_w[:, self._ee_body_idxs, :3]
		threshold_sq = float(grasp_dist_thresh) * float(grasp_dist_thresh)

		reached_flags: list[torch.Tensor] = []
		for obj_idx in range(1, self._num_objects + 1):
			obj = env.scene[f"object_{obj_idx}"]
			obj_pos = obj.data.root_pos_w[:, :3]
			dist_sq = torch.sum((ee_pos - obj_pos.unsqueeze(1)) ** 2, dim=2)
			reached_flags.append((dist_sq <= threshold_sq).any(dim=1))

		reached = torch.stack(reached_flags, dim=1)
		newly_reached = reached & (~self._counted)
		self._counted |= reached

		return newly_reached.sum(dim=1).to(torch.float32) * float(reward_per_object)
