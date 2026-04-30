from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.managers.manager_base import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CrossXMulti(ManagerTermBase):
    """Give distributed rewards between x-threshold segments based on progress."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._initialized = False

        # reward buffers
        self._thresholds = None
        self._rewards = None
        self._segment_starts = None
        self._segment_lengths = None
        self._prev_cumulative_reward = None
        self._max_robot_x = None
        self._triggered = None
        self._prev_crossed = None
        self._asset_name = None

        # visual assets
        self._visual_spawned = False
        self._visual_prim_paths = []
        self._last_visual_update_step = -1

    def _init_buffers(
        self,
        x_thresholds: Sequence[float],
        rewards: Sequence[float],
        start_x: float,
    ):
        if len(x_thresholds) != len(rewards):
            raise ValueError(
                f"x_thresholds and rewards must have the same length, "
                f"got {len(x_thresholds)} and {len(rewards)}."
            )

        if len(x_thresholds) == 0:
            raise ValueError("x_thresholds cannot be empty.")

        self._thresholds = torch.tensor(
            x_thresholds, device=self._env.device, dtype=torch.float32
        )
        self._rewards = torch.tensor(
            rewards, device=self._env.device, dtype=torch.float32
        )

        # Segment i is [segment_starts[i], thresholds[i]] and accumulates rewards[i].
        self._segment_starts = torch.cat(
            [
                torch.tensor([start_x], device=self._env.device, dtype=torch.float32),
                self._thresholds[:-1],
            ]
        )
        self._segment_lengths = self._thresholds - self._segment_starts
        if torch.any(self._segment_lengths <= 0):
            raise ValueError(
                "x_thresholds must be strictly increasing and greater than start_x."
            )

        num_checkpoints = len(x_thresholds)

        self._triggered = torch.zeros(
            (self._env.num_envs, num_checkpoints),
            device=self._env.device,
            dtype=torch.bool,
        )
        self._prev_crossed = torch.zeros(
            (self._env.num_envs, num_checkpoints),
            device=self._env.device,
            dtype=torch.bool,
        )
        self._prev_cumulative_reward = None
        self._max_robot_x = None

        self._initialized = True

    def _compute_cumulative_reward(self, robot_x: torch.Tensor) -> torch.Tensor:
        # Sum segment-wise linear progress, each segment saturates to its own max reward.
        progress = (robot_x.unsqueeze(1) - self._segment_starts.unsqueeze(0)) / self._segment_lengths.unsqueeze(0)
        progress = progress.clamp(min=0.0, max=1.0)
        return (progress * self._rewards.unsqueeze(0)).sum(dim=1)

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
            print(f"[CrossXMulti] Failed to set color for {prim_path}: {e}")

    def _spawn_threshold_assets_once(
        self,
        x_thresholds: Sequence[float],
        parent_prim_path: str = "/World/Visuals/CrossXMulti",
        line_length_y: float = 4.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.5,
        color_default: tuple[float, float, float] = (1.0, 0.2, 0.2),
    ):
        """Spawn thin cuboids as visual threshold lines.
        """
        if self._visual_spawned:
            return

        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            if not stage.GetPrimAtPath(parent_prim_path).IsValid():
                from pxr import UsdGeom
                UsdGeom.Xform.Define(stage, parent_prim_path)
        except Exception:
            pass

        self._visual_prim_paths = []

        for env_id in range(self._env.num_envs):
            env_paths = []

            for i, x_th in enumerate(x_thresholds):
                x_world = float(x_th)
                y_world = 0.0
                z_world = 2.0 + line_height_z * 0.5

                prim_path = f"{parent_prim_path}/env_{env_id}_threshold_{i}"

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
                        float(x_world),
                        float(y_world),
                        float(z_world),
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
        if not self._visual_spawned:
            return

        for env_id in range(self._env.num_envs):
            for i in range(len(self._thresholds)):
                prim_path = self._visual_prim_paths[env_id][i]
                color = color_triggered if self._triggered[env_id, i] else color_default
                self._set_prim_color(prim_path, color)

    def reset(self, env_ids=None):
        if not self._initialized:
            return

        if env_ids is None:
            self._triggered.fill_(False)
            self._prev_crossed.fill_(False)
            if self._asset_name is not None:
                robot_x = self._env.scene[self._asset_name].data.root_pos_w[:, 0]
                if self._max_robot_x is None:
                    self._max_robot_x = robot_x.clone()
                else:
                    self._max_robot_x.copy_(robot_x)
                cumulative = self._compute_cumulative_reward(self._max_robot_x)
                if self._prev_cumulative_reward is None:
                    self._prev_cumulative_reward = cumulative.clone()
                else:
                    self._prev_cumulative_reward.copy_(cumulative)
            elif self._prev_cumulative_reward is not None:
                self._prev_cumulative_reward.zero_()
                if self._max_robot_x is not None:
                    self._max_robot_x.zero_()
        else:
            self._triggered[env_ids] = False
            self._prev_crossed[env_ids] = False
            if self._asset_name is not None:
                robot_x = self._env.scene[self._asset_name].data.root_pos_w[:, 0]
                if self._max_robot_x is None:
                    self._max_robot_x = robot_x.clone()
                self._max_robot_x[env_ids] = robot_x[env_ids]
                cumulative = self._compute_cumulative_reward(self._max_robot_x)
                if self._prev_cumulative_reward is None:
                    self._prev_cumulative_reward = cumulative.clone()
                else:
                    self._prev_cumulative_reward[env_ids] = cumulative[env_ids]
            elif self._prev_cumulative_reward is not None:
                self._prev_cumulative_reward[env_ids] = 0.0
                if self._max_robot_x is not None:
                    self._max_robot_x[env_ids] = 0.0

        if self._visual_spawned:
            self._update_threshold_asset_colors()

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg,
        x_thresholds,
        rewards,
        start_x: float = -141.0,
        debug: bool = False,
        visual_assets: bool = False,
        visual_update_interval: int = 10,
        parent_prim_path: str = "/World/Visuals/CrossXMulti",
        line_length_y: float = 20.0,
        line_thickness_x: float = 0.02,
        line_height_z: float = 0.5,
    ) -> torch.Tensor:
        # reward buffer init
        if not self._initialized:
            self._init_buffers(x_thresholds, rewards, start_x=start_x)

        if self._asset_name is None:
            self._asset_name = asset_cfg.name

        if visual_assets and not self._visual_spawned:
            self._spawn_threshold_assets_once(
                x_thresholds=x_thresholds,
                parent_prim_path=parent_prim_path,
                line_length_y=line_length_y,
                line_thickness_x=line_thickness_x,
                line_height_z=line_height_z,
            )

        robot = env.scene[asset_cfg.name]

        robot_x = robot.data.root_pos_w[:, 0]  # [E]
        if self._max_robot_x is None:
            self._max_robot_x = robot_x.clone()
        prev_max_robot_x = self._max_robot_x.clone()
        self._max_robot_x = torch.maximum(self._max_robot_x, robot_x)
        max_robot_x = self._max_robot_x

        cumulative_reward = self._compute_cumulative_reward(max_robot_x)

        if self._prev_cumulative_reward is None:
            # Avoid giving a spike on the very first frame.
            reward = torch.zeros_like(cumulative_reward)
            self._prev_cumulative_reward = cumulative_reward.clone()
        else:
            reward = cumulative_reward - self._prev_cumulative_reward
            no_forward_progress = max_robot_x <= (prev_max_robot_x + 1e-6)
            reward = torch.where(no_forward_progress, torch.zeros_like(reward), reward)
            reward = reward.clamp(min=0.0)
            self._prev_cumulative_reward.copy_(cumulative_reward)

        # Safety guard: never return negative reward.
        reward = torch.where(reward < 0.0, torch.zeros_like(reward), reward)

        crossed = max_robot_x.unsqueeze(1) > self._thresholds.unsqueeze(0)
        newly_crossed = crossed & (~self._prev_crossed) & (~self._triggered)
        self._triggered |= newly_crossed
        self._prev_crossed = crossed

        if debug:
            print(
                f"[CrossXMulti] x={robot_x[0].item():.3f}, "
                f"max_x={max_robot_x[0].item():.3f}, "
                f"cumulative={cumulative_reward[0].item():.3f}, "
                f"reward={reward[0].item():.3f}, "
                f"newly_crossed={newly_crossed[0].tolist()}"
            )

        if visual_assets and self._visual_spawned:
            step = getattr(env, "common_step_counter", 0)
            if step != self._last_visual_update_step and step % visual_update_interval == 0:
                self._update_threshold_asset_colors()
                self._last_visual_update_step = step

        return reward.clamp(min=0.0)
