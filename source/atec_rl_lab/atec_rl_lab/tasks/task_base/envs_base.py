# Created by skywoodsz on 2026/03/03.

import torch
from isaaclab.envs import ManagerBasedRLEnv

class BaseRLEnv(ManagerBasedRLEnv):
    def step(self, action: torch.Tensor):
        elapsed_time = self.episode_length_buf.to(torch.float32) * float(self.step_dt)

        obs, reward, terminated, truncated, info = super().step(action)

        info["Elapsed_Time"] = elapsed_time
        info["Step_dt"] = float(self.step_dt)
        info["Episode_Length_s"] = float(self.max_episode_length_s)

        return obs, reward, terminated, truncated, info
