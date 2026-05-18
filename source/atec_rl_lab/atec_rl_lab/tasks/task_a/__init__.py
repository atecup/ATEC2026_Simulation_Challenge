import gymnasium as gym

gym.register(
    id = "ATEC-TaskA-G1",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvG1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron1Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron1Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron2ALegged",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2ALeggedCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-Tron2AWheel",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvTron2AWheelCfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2Piper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2Cfg"
    },
)

gym.register(
    id = "ATEC-TaskA-B2wPiper",
    entry_point="atec_rl_lab.tasks.task_base.envs_base:BaseRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:TaskAEnvB2WCfg"
    },
)

from .env_cfg import TaskAEnvCfg, TaskAEnvB2Cfg, TaskAEnvTron2ALeggedCfg, TaskAEnvTron2AWheelCfg
__all__ = ['TaskAEnvCfg', 'TaskAEnvB2Cfg', 'TaskAEnvTron2ALeggedCfg', 'TaskAEnvTron2AWheelCfg']
