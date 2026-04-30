from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.sim as sim_utils
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardCrossX(ManagerTermBase):
    """One-time reward when robot crosses x threshold(s).
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._initialized = False
        self._reward_given = None

        # visual assets
        self._visual_spawned = False
        self._visual_prim_paths = []
        self._last_visual_update_step = -1

    def _init_buffers(self, num_thresholds: int = 1):
        if self._initialized:
            return

        self._reward_given = torch.zeros(
            (self._env.num_envs, num_thresholds),
            dtype=torch.bool,
            device=self._env.device,
        )
        self._initialized = True

    def _set_prim_color(self, prim_path: str, color: tuple[float, float, float]):
        """Set display color of a spawned cuboid prim."""
        try:
            import omni.usd
            from pxr import UsdGeom, Vt

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return

            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr()
            gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
        except Exception as e:
            print(f"[RewardCrossX] Failed to set color for {prim_path}: {e}")

    def _spawn_threshold_assets_once(
        self,
        thresholds: Sequence[float],
        parent_prim_path: str = "/World/Visuals/RewardCrossX",
        line_length_y: float = 10.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.25,
        color_default: tuple[float, float, float] = (1.0, 0.2, 0.2),
    ):
        """Spawn one thin cuboid line for each threshold in each env.
        """
        if self._visual_spawned:
            return

        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath(parent_prim_path).IsValid():
                UsdGeom.Xform.Define(stage, parent_prim_path)
        except Exception:
            pass

        self._visual_prim_paths = []

        for env_id in range(self._env.num_envs):
            env_paths = []
            for th_idx, th in enumerate(thresholds):
                prim_path = f"{parent_prim_path}/env_{env_id}_threshold_{th_idx}"

                cfg = sim_utils.CuboidCfg(
                    size=(line_thickness_x, line_length_y, line_height_z),
                    visual_material=None,
                    collision_props=None,
                    rigid_props=None,
                    mass_props=None,
                )

                cfg.func(
                    prim_path=prim_path,
                    cfg=cfg,
                    translation=(
                        float(th),
                        0.0,
                        float(3.0 + line_height_z * 0.5),
                    ),
                )

                self._set_prim_color(prim_path, color_default)
                env_paths.append(prim_path)

            self._visual_prim_paths.append(env_paths)

        self._visual_spawned = True

    def _update_threshold_asset_colors(
        self,
        color_default: tuple[float, float, float] = (1.0, 0.2, 0.2),
        color_triggered: tuple[float, float, float] = (0.2, 1.0, 0.2),
    ):
        """Update line color based on per-threshold reward status."""
        if not self._visual_spawned:
            return

        for env_id, env_paths in enumerate(self._visual_prim_paths):
            for th_idx, prim_path in enumerate(env_paths):
                color = color_triggered if self._reward_given[env_id, th_idx] else color_default
                self._set_prim_color(prim_path, color)

    def reset(self, env_ids=None):
        if not self._initialized:
            return

        if env_ids is None:
            self._reward_given.fill_(False)
        else:
            self._reward_given[env_ids] = False

        if self._visual_spawned:
            self._update_threshold_asset_colors()

    def _normalize_threshold_reward_params(
        self,
        thresholds: Sequence[float] | None,
        reward_values: Sequence[float] | None,
        threshold: float | Sequence[float],
        reward_value: float | Sequence[float],
        threshold_2: float | None,
        reward_value_2: float,
    ) -> tuple[list[float], list[float]]:
        """Normalize scalar/list args into float lists.
        """

        def _is_sequence(v) -> bool:
            return isinstance(v, Sequence) and not isinstance(v, (str, bytes))

        if thresholds is not None or reward_values is not None:
            if thresholds is None or reward_values is None:
                raise ValueError("thresholds and reward_values must be provided together.")
            thresholds_list = [float(x) for x in thresholds]
            reward_values_list = [float(x) for x in reward_values]
            if _is_sequence(threshold) or _is_sequence(reward_value) or threshold_2 is not None:
                raise ValueError(
                    "Do not mix thresholds/reward_values with threshold/reward_value/threshold_2."
                )
        else:
            th_is_seq = _is_sequence(threshold)
            rew_is_seq = _is_sequence(reward_value)
            if th_is_seq != rew_is_seq:
                raise ValueError(
                    "threshold and reward_value must both be scalar or both be list."
                )

            if th_is_seq:
                if threshold_2 is not None:
                    raise ValueError("threshold_2 cannot be used when threshold is a list.")
                thresholds_list = [float(x) for x in threshold]
                reward_values_list = [float(x) for x in reward_value]
            else:
                thresholds_list = [float(threshold)]
                reward_values_list = [float(reward_value)]
                if threshold_2 is not None:
                    thresholds_list.append(float(threshold_2))
                    reward_values_list.append(float(reward_value_2))

        if len(thresholds_list) == 0:
            raise ValueError("At least one threshold is required.")
        if len(thresholds_list) != len(reward_values_list):
            raise ValueError(
                "thresholds and reward_values must have same length, "
                f"got {len(thresholds_list)} and {len(reward_values_list)}."
            )
        for i in range(1, len(thresholds_list)):
            if thresholds_list[i] <= thresholds_list[i - 1]:
                raise ValueError("thresholds must be strictly increasing.")

        return thresholds_list, reward_values_list

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg,
        threshold: float | Sequence[float] = 0.6,
        reward_value: float | Sequence[float] = 24.0,
        threshold_2: float | None = None,
        reward_value_2: float = 0.0,
        thresholds: Sequence[float] | None = None,
        reward_values: Sequence[float] | None = None,
        debug: bool = False,
        visual_assets: bool = False,
        visual_update_interval: int = 10,
        parent_prim_path: str = "/World/Visuals/RewardCrossX",
        line_length_y: float = 10.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.5,
    ) -> torch.Tensor:
        thresholds, reward_values = self._normalize_threshold_reward_params(
            thresholds=thresholds,
            reward_values=reward_values,
            threshold=threshold,
            reward_value=reward_value,
            threshold_2=threshold_2,
            reward_value_2=reward_value_2,
        )

        num_thresholds = len(thresholds)

        if not self._initialized:
            self._init_buffers(num_thresholds=num_thresholds)
        elif self._reward_given.shape[1] != num_thresholds:
            raise ValueError(
                "RewardCrossX threshold count changed after initialization. "
                f"Expected {self._reward_given.shape[1]}, got {num_thresholds}."
            )

        if visual_assets and not self._visual_spawned:
            self._spawn_threshold_assets_once(
                thresholds=thresholds,
                parent_prim_path=parent_prim_path,
                line_length_y=line_length_y,
                line_thickness_x=line_thickness_x,
                line_height_z=line_height_z,
            )
        elif visual_assets and self._visual_spawned and self._visual_prim_paths:
            if len(self._visual_prim_paths[0]) != num_thresholds:
                raise ValueError(
                    "RewardCrossX visual threshold count changed after spawn. "
                    f"Expected {len(self._visual_prim_paths[0])}, got {num_thresholds}."
                )

        robot = env.scene[asset_cfg.name]

        root_pos_x = robot.data.root_pos_w[:, 0]
        thresholds_t = torch.tensor(thresholds, device=root_pos_x.device, dtype=root_pos_x.dtype)
        reward_values_t = torch.tensor(reward_values, device=root_pos_x.device, dtype=root_pos_x.dtype)

        crossed = root_pos_x.unsqueeze(1) > thresholds_t.unsqueeze(0)
        trigger = crossed & (~self._reward_given)
        self._reward_given |= crossed

        reward = (trigger.float() * reward_values_t.unsqueeze(0)).sum(dim=1)

        if debug:
            print(
                f"[RewardCrossX] x={root_pos_x[0].item():.3f}, "
                f"crossed={crossed[0].tolist()}, "
                f"trigger={trigger[0].tolist()}, "
                f"reward={reward[0].item():.3f}"
            )

        if visual_assets and self._visual_spawned:
            step = getattr(env, "common_step_counter", 0)
            if step != self._last_visual_update_step and step % visual_update_interval == 0:
                self._update_threshold_asset_colors()
                self._last_visual_update_step = step

        return reward


class RewardBoxXInRange(ManagerTermBase):
    """Give reward when the target box x-position is within one or more x-ranges."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._reward_given = torch.zeros(self._env.num_envs, dtype=torch.bool, device=self._env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._reward_given.fill_(False)
        else:
            self._reward_given[env_ids] = False

    def _normalize_ranges(
        self,
        x_min: float | Sequence[float],
        x_max: float | Sequence[float],
    ) -> tuple[list[float], list[float]]:
        is_min_seq = isinstance(x_min, Sequence) and not isinstance(x_min, (str, bytes))
        is_max_seq = isinstance(x_max, Sequence) and not isinstance(x_max, (str, bytes))

        if is_min_seq != is_max_seq:
            raise ValueError("x_min and x_max must both be scalar or both be sequences.")

        if is_min_seq:
            x_min_list = [float(v) for v in x_min]
            x_max_list = [float(v) for v in x_max]
            if len(x_min_list) == 0:
                raise ValueError("At least one x-range is required.")
            if len(x_min_list) != len(x_max_list):
                raise ValueError("x_min and x_max sequences must have the same length.")
        else:
            x_min_list = [float(x_min)]
            x_max_list = [float(x_max)]

        for mn, mx in zip(x_min_list, x_max_list):
            if mn > mx:
                raise ValueError(f"Invalid x-range: x_min ({mn}) must be <= x_max ({mx}).")

        return x_min_list, x_max_list

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg,
        x_min: float | Sequence[float] = -1.0,
        x_max: float | Sequence[float] = 1.0,
        reward_value: float = 14.0,
        one_time: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        box = env.scene[asset_cfg.name]
        box_x = box.data.root_pos_w[:, 0]

        x_min_list, x_max_list = self._normalize_ranges(x_min=x_min, x_max=x_max)

        in_any_range = torch.zeros_like(box_x, dtype=torch.bool)
        for mn, mx in zip(x_min_list, x_max_list):
            in_any_range |= (box_x >= mn) & (box_x <= mx)

        if one_time:
            trigger = in_any_range & (~self._reward_given)
            self._reward_given |= in_any_range
            reward = trigger.to(box_x.dtype) * float(reward_value)
        else:
            reward = in_any_range.to(box_x.dtype) * float(reward_value)

        if debug:
            print(
                f"[RewardBoxXInRange] box_x={box_x[0].item():.3f}, "
                f"in_any_range={bool(in_any_range[0].item())}, "
                f"ranges={list(zip(x_min_list, x_max_list))}, "
                f"reward={reward[0].item():.3f}"
            )

        return reward

