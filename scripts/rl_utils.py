import torch
import isaaclab.utils.math as math_utils

def camera_follow(env, robot_name: str = "robot", env_index: int = 0, alpha: float = 0.15):
    unwrapped = env.unwrapped

    if not hasattr(unwrapped, "viewport_camera_controller"):
        return

    try:
        robot = unwrapped.scene[robot_name]
    except KeyError as e:
        raise KeyError(
            f"Robot asset '{robot_name}' not found in env.unwrapped.scene."
        ) from e

    device = unwrapped.device

    robot_pos = robot.data.root_pos_w[env_index]
    robot_quat = robot.data.root_quat_w[env_index]

    camera_offset = torch.tensor([-6.0, 0.0, 0.8], dtype=torch.float32, device=device)

    target_camera_pos = math_utils.transform_points(
        camera_offset.unsqueeze(0),
        pos=robot_pos.unsqueeze(0),
        quat=robot_quat.unsqueeze(0),
    ).squeeze(0)

    target_camera_pos[2] = torch.clamp(target_camera_pos[2], min=0.2)

    if not hasattr(camera_follow, "_smooth_pos"):
        camera_follow._smooth_pos = {}

    if env_index not in camera_follow._smooth_pos:
        camera_follow._smooth_pos[env_index] = target_camera_pos.clone()

    smooth_camera_pos = camera_follow._smooth_pos[env_index]
    smooth_camera_pos = (1.0 - alpha) * smooth_camera_pos + alpha * target_camera_pos
    camera_follow._smooth_pos[env_index] = smooth_camera_pos

    unwrapped.viewport_camera_controller.set_view_env_index(env_index=env_index)
    unwrapped.viewport_camera_controller.update_view_location(
        eye=smooth_camera_pos.detach().cpu().numpy(),
        lookat=robot_pos.detach().cpu().numpy(),
    )