# Created by skywoodsz on 2026/01/28.
import argparse
import itertools
import torch
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="View ATEC Task D.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.envs import ManagerBasedRLEnv

from atec_rl_lab.tasks.task_d import TaskDEnvB2Cfg


def main():
    env_cfg = TaskDEnvB2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(env_cfg)

    for name, articulation in env.scene.articulations.items():
        print("-" * 100)
        print("Robot name:", name)
        print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
        print("Joints:", articulation.num_joints, "->", articulation.joint_names)
        articulation.set_joint_position_target(articulation.data.default_joint_pos)

    action_space = env.action_space
    obs, info = env.reset()

    for i in itertools.count():
        if not simulation_app.is_running():
            break
        action = torch.zeros(action_space.shape, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        
        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=env_ids)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
