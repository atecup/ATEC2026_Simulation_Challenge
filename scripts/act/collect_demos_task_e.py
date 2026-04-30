"""Scripted oracle data collection for Task E (pick-and-place).

Usage
-----
# NOTE: only for object 3 now, if you want to pick objects 1 and 2, please modify the distance accordingly.
python scripts/act/collect_demos_task_e.py --pick_objects 3 --num_demos 50 --headless

"""

import argparse
import os
import sys

# sys.path.insert(0, os.path.dirname(__file__))

from isaaclab.app import AppLauncher
from cli_args import add_collect_demo_args

parser = argparse.ArgumentParser(description="Collect Task E demonstrations for ACT.")
add_collect_demo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.save_video or args_cli.save_images:
    args_cli.enable_cameras = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

import h5py
import json
import numpy as np

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg
import isaaclab.sim as sim_utils

from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg
from atec_rl_lab.utils import CartesianController

from task_e.config import (
    EE_BODY_NAME, ARM_JOINT_NAMES, GRIPPER_JOINT_NAMES,
    ACT_STIFFNESS, ACT_DAMPING, ACT_EFFORT_LIMIT, ACT_VEL_LIMIT,
    CAM_H, CAM_W, CAM_POS, CAM_ROT,
)
from task_e.collector import check_objects_in_basket, collect_one_demo


def build_env(pick_objects: list[int], need_camera: bool) -> ManagerBasedRLEnv:
    import time
    cfg = TaskEEnvPiperCfg()
    cfg.seed              = int(time.time_ns() % (2**31))   # random seed each call
    cfg.scene.num_envs    = 1
    cfg.episode_length_s  = 40.0 * len(pick_objects) + 10.0
    cfg.scene.robot.actuators["default"] = ImplicitActuatorCfg(
        joint_names_expr=[".*"],
        effort_limit=ACT_EFFORT_LIMIT,
        velocity_limit=ACT_VEL_LIMIT,
        stiffness=ACT_STIFFNESS,
        damping=ACT_DAMPING,
    )
    # if need_camera:
    #     cfg.scene.video_cam = CameraCfg(
    #         prim_path="{ENV_REGEX_NS}/video_cam",
    #         update_period=0.0,
    #         height=CAM_H, width=CAM_W,
    #         data_types=["rgb"],
    #         spawn=sim_utils.PinholeCameraCfg(
    #             focal_length=24.0, focus_distance=400.0,
    #             horizontal_aperture=20.955, clipping_range=(0.1, 100.0),
    #         ),
    #         offset=CameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
    #     )
    return ManagerBasedRLEnv(cfg)


def init_output(output_dir: str) -> tuple[str, str]:
    """Create output directory, wipe any existing trajectory.hdf5, write JSON metadata."""
    os.makedirs(output_dir, exist_ok=True)
    traj_path = os.path.join(output_dir, "trajectory.hdf5")
    json_path = os.path.join(output_dir, "trajectory.json")
    with h5py.File(traj_path, "w"):   # truncate / create fresh
        pass
    with open(json_path, "w") as fh:
        json.dump({"env_info": {"env_kwargs": {"control_mode": "pd_joint_pos"}}}, fh)
    return traj_path, json_path


def save_traj(traj_path: str, traj_idx: int, data: dict,
              save_images: bool) -> None:
    """Append one trajectory group to the consolidated HDF5."""
    with h5py.File(traj_path, "a") as f:
        grp = f.create_group(f"traj_{traj_idx}")
        grp.create_dataset("obs",     data=data["qpos"],    compression="gzip")
        grp.create_dataset("actions", data=data["action"],  compression="gzip")
        grp.create_dataset("qvel",    data=data["qvel"],    compression="gzip")
        grp.create_dataset("ee_pos",  data=data["ee_pos"],  compression="gzip")
        grp.create_dataset("ee_quat", data=data["ee_quat"], compression="gzip")
        if save_images and "frames" in data:
            grp.create_group("images").create_dataset(
                "rgb", data=data["frames"], compression="gzip"
            )


def main() -> None:
    pick_objects = sorted(set(args_cli.pick_objects))
    need_camera = args_cli.save_video or args_cli.save_images

    env    = build_env(pick_objects, need_camera)
    dev    = env.unwrapped.device
    camera = env.unwrapped.scene["video_cam"] if need_camera else None

    robot = env.unwrapped.scene.articulations["robot"]
    arm_ids,     _ = robot.find_joints(ARM_JOINT_NAMES)
    gripper_ids, _ = robot.find_joints(GRIPPER_JOINT_NAMES)
    ik_ctrl = CartesianController(
        robot=robot, ee_body_name=EE_BODY_NAME,
        arm_joint_names=ARM_JOINT_NAMES,
        num_envs=1, device=dev,
        command_type="pose",
        lambda_val=0.05,
        max_joint_delta=0.2,
    )
    default_jpos = robot.data.default_joint_pos.clone()

    video_dir = None
    imageio   = None
    if args_cli.save_video:
        video_dir = args_cli.video_dir or os.path.join(args_cli.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        import imageio as _io
        imageio = _io

    traj_path, _ = init_output(args_cli.output_dir)
    rng = np.random.default_rng()   # unseeded → different positions every run

    n_ok = 0
    attempt = 0
    while n_ok < args_cli.num_demos:
        attempt += 1
        print(f"\n[INFO] Demo {n_ok + 1}/{args_cli.num_demos}  (attempt {attempt})")

        data = collect_one_demo(
            env, robot, ik_ctrl,
            arm_ids, gripper_ids,
            pick_objects, dev,
            default_jpos=default_jpos,
            rng=rng,
            camera=camera,
        )
        if data is None:
            print("[WARN] Early termination — skipping.")
            continue

        if args_cli.only_success and not check_objects_in_basket(env, pick_objects):
            print("[WARN] Objects not in basket — skipping (--only_success).")
            continue

        save_traj(traj_path, n_ok, data, args_cli.save_images)

        T     = len(data["qpos"])
        notes = [f"{T} steps"]
        if args_cli.save_video and "frames" in data:
            vp = os.path.join(video_dir, f"demo_{n_ok:04d}.mp4")
            imageio.mimwrite(vp, data["frames"], fps=50, quality=7)
            notes.append(f"video → {vp}")
        if args_cli.save_images and "frames" in data:
            notes.append("images saved")
        print(f"[INFO] traj_{n_ok}: {', '.join(notes)}")
        n_ok += 1

    print(f"\n[INFO] Collected {n_ok} demos → {traj_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
