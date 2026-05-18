import gymnasium as gym

gym.register(
    id = "ATEC-TaskB-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskB-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskB-Tron2ALegged",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "ATEC-TaskB-Tron2AWheel",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "ATEC-TaskB-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvB2Cfg"
    },
)

gym.register(
    id = "ATEC-TaskB-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskBEnvB2WCfg"
    },
)

from .env_cfg import TaskBEnvCfg, TaskBEnvB2Cfg, TaskBEnvTron2ALeggedCfg, TaskBEnvTron2AWheelCfg
__all__ = ['TaskBEnvCfg', 'TaskBEnvB2Cfg', 'TaskBEnvTron2ALeggedCfg', 'TaskBEnvTron2AWheelCfg']
