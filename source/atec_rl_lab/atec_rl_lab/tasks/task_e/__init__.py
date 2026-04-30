import gymnasium as gym

gym.register(
    id="ATEC-TaskE-Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskEEnvPiperCfg",
    },
)

from .env_cfg import TaskEEnvPiperCfg

__all__ = ["TaskEEnvPiperCfg"]
