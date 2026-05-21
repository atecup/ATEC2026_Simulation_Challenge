# Created by skywoodsz on 2026/02/07.

import argparse
import os
import time
import json

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow
from atec_rl_lab.tasks.task_base.action_base import apply_safe_action_spec

from demo.solution import AlgSolution
solution = AlgSolution()

def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    # TODO: simulate getting action spec from jason string (e.g. from a file or network)
    # action_spec = solution.get_action_spec() if hasattr(solution, "get_action_spec") else None
    # action_spec_json = json.dumps(action_spec)

    # New Feature: apply safe action spec to env config (e.g. for scaling/clipping actions from your solution)
    env_cfg = apply_safe_action_spec(env_cfg, action_spec_json)
    
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video:
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            # ===== Your controller goes here =====
            resp = solution.predicts(obs, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            if not is_task_e:
                camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")

            done = (terminated.item() or truncated.item())
            if done:
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
