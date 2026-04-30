"""
Cartesian-space IK interface for robot end-effector control.

Example usage::

    from atec_rl_lab.utils import CartesianController

    ctrl = CartesianController(
        robot=env.scene.articulations["robot"],
        ee_body_name="gripper_base",
        arm_joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        num_envs=env.num_envs,
        device=env.device,
    )
    ctrl.reset()

    # In simulation loop:
    arm_jpos_des = ctrl.compute(ee_pos_w, ee_quat_w)  # world-frame target
    # or
    arm_jpos_des = ctrl.compute_base(ee_pos_b, ee_quat_b)  # base-frame target
"""

import torch
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import subtract_frame_transforms, quat_inv, matrix_from_quat, quat_rotate_inverse


class CartesianController:
    """
    Cartesian-space controller for a robot arm.

    Converts desired end-effector poses in world frame to arm joint
    position targets using Differential Inverse Kinematics.
    Also supports directly consuming base-frame targets via ``compute_base``.

    Parameters
    ----------
    robot : Articulation
        The robot articulation object from the Isaac Lab scene.
    ee_body_name : str
        Name of the end-effector body/link (e.g. "gripper_base").
    arm_joint_names : list[str]
        Names of the arm joints to control (excludes gripper fingers).
    num_envs : int
        Number of parallel environments.
    device : str
        Torch device string, e.g. "cuda:0".
    command_type : str
        ``"pose"`` — control both position and orientation (default).
        ``"position"`` — control position only; orientation floats freely.
        Use ``"position"`` when the arm starts far from the desired orientation
        or when orientation doesn't need to be constrained.
    lambda_val : float
        Damping factor for Damped Least Squares IK (default 0.1).
        Increase for smoother motion near singularities.
    max_joint_delta : float
        Maximum joint position change per simulation step (radians).
        Limits the IK step size to prevent oscillation. Default 0.05 rad/step
        (≈2.5 rad/s at 50 Hz = ~143 deg/s). Set to None to disable clamping.
    """

    def __init__(
        self,
        robot: Articulation,
        ee_body_name: str,
        arm_joint_names: list[str],
        num_envs: int,
        device: str,
        command_type: str = "pose",
        lambda_val: float = 0.1,
        max_joint_delta: float = 0.05,
    ):
        self.robot = robot
        self.device = device
        self.command_type = command_type
        self.max_joint_delta = max_joint_delta

        # Resolve body and joint indices 
        body_ids, body_names = robot.find_bodies(ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body matching '{ee_body_name}', "
                f"found {len(body_ids)}: {body_names}"
            )
        self.ee_idx: int = body_ids[0]

        self.arm_ids, arm_names = robot.find_joints(arm_joint_names)
        if len(self.arm_ids) == 0:
            raise ValueError(
                f"No joints found matching {arm_joint_names}. "
                f"Available joints: {robot.joint_names}"
            )

        if robot.is_fixed_base:
            # Fixed-base: Jacobian rows are body_idx - 1 (root body excluded)
            self._jacobi_body_idx = self.ee_idx - 1
            self._jacobi_joint_ids = list(self.arm_ids)
        else:
            # Floating-base: root body IS included, joint columns shifted +6
            self._jacobi_body_idx = self.ee_idx
            self._jacobi_joint_ids = [i + 6 for i in self.arm_ids]

        # Build IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type=command_type,
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": lambda_val},
        )
        self.ik_ctrl = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

        # print(
        #     f"[CartesianController] ee='{ee_body_name}' (idx={self.ee_idx}), "
        #     f"arm_joints={arm_names}, fixed_base={robot.is_fixed_base}, "
        #     f"command_type='{command_type}'"
        # )

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Reset the IK state and seed it from the robot's current EE pose.

        Call this after every ``env.reset()`` to avoid stale IK state.

        Parameters
        ----------
        env_ids : torch.Tensor, optional
            Indices of environments to reset.  If None, resets all.
        """
        self.ik_ctrl.reset(env_ids)

        root_pose_w = self.robot.data.root_pose_w
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_idx]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:],
            ee_pose_w[:, :3], ee_pose_w[:, 3:],
        )
        if self.command_type == "position":
            self.ik_ctrl.set_command(ee_pos_b, ee_quat=ee_quat_b)
        else:
            self.ik_ctrl.set_command(torch.cat([ee_pos_b, ee_quat_b], dim=-1))

    def compute(
        self,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute desired arm joint positions from a world-frame EE target.

        Parameters
        ----------
        ee_pos_w : torch.Tensor
            Desired EE position in world frame, shape (num_envs, 3).
        ee_quat_w : torch.Tensor, optional
            Desired EE orientation (w, x, y, z) in world frame,
            shape (num_envs, 4).  Required when command_type="pose";
            ignored when command_type="position".

        Returns
        -------
        arm_jpos_des : torch.Tensor
            Desired joint positions for the arm joints,
            shape (num_envs, n_arm_joints).
        """
        root_pose_w = self.robot.data.root_pose_w

        if self.command_type == "position":
            ee_pos_des_b = quat_rotate_inverse(
                root_pose_w[:, 3:], ee_pos_w - root_pose_w[:, :3]
            )
            return self.compute_base(ee_pos_des_b, ee_quat_b=None)

        if ee_quat_w is None:
            raise ValueError("ee_quat_w is required when command_type='pose'")

        ee_pos_des_b, ee_quat_des_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:],
            ee_pos_w, ee_quat_w,
        )
        return self.compute_base(ee_pos_des_b, ee_quat_b=ee_quat_des_b)

    def compute_base(
        self,
        ee_pos_b: torch.Tensor,
        ee_quat_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute desired arm joint positions from a base-frame EE target.

        Parameters
        ----------
        ee_pos_b : torch.Tensor
            Desired EE position in robot base frame, shape (num_envs, 3).
        ee_quat_b : torch.Tensor, optional
            Desired EE orientation (w, x, y, z) in robot base frame,
            shape (num_envs, 4). Required when command_type="pose";
            ignored when command_type="position".

        Returns
        -------
        arm_jpos_des : torch.Tensor
            Desired joint positions for the arm joints,
            shape (num_envs, n_arm_joints).
        """
        root_pose_w = self.robot.data.root_pose_w

        if self.command_type == "position":
            ee_pose_w_cur = self.robot.data.body_pose_w[:, self.ee_idx]
            _, ee_quat_b_now = subtract_frame_transforms(
                root_pose_w[:, :3], root_pose_w[:, 3:],
                ee_pose_w_cur[:, :3], ee_pose_w_cur[:, 3:],
            )
            self.ik_ctrl.set_command(ee_pos_b, ee_quat=ee_quat_b_now)
        else:
            if ee_quat_b is None:
                raise ValueError("ee_quat_b is required when command_type='pose'")
            self.ik_ctrl.set_command(torch.cat([ee_pos_b, ee_quat_b], dim=-1))

        return self._solve_ik_with_current_state(root_pose_w)

    def _solve_ik_with_current_state(self, root_pose_w: torch.Tensor) -> torch.Tensor:
        """Run IK solve from current robot state after command has been set."""
        # Retrieve Jacobian (world frame) and rotate to base frame
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ]
        base_rot_mat = matrix_from_quat(quat_inv(root_pose_w[:, 3:]))
        jacobian[:, :3] = torch.bmm(base_rot_mat, jacobian[:, :3])
        jacobian[:, 3:] = torch.bmm(base_rot_mat, jacobian[:, 3:])

        # Current EE pose in base frame
        ee_pose_w_cur = self.robot.data.body_pose_w[:, self.ee_idx]
        ee_pos_b_cur, ee_quat_b_cur = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:],
            ee_pose_w_cur[:, :3], ee_pose_w_cur[:, 3:],
        )

        arm_jpos_cur = self.robot.data.joint_pos[:, self.arm_ids]
        arm_jpos_des = self.ik_ctrl.compute(
            ee_pos_b_cur, ee_quat_b_cur, jacobian, arm_jpos_cur
        )

        # Clamp per-step delta to prevent oscillation from large IK steps
        if self.max_joint_delta is not None:
            delta = arm_jpos_des - arm_jpos_cur
            delta = torch.clamp(delta, -self.max_joint_delta, self.max_joint_delta)
            arm_jpos_des = arm_jpos_cur + delta

        return arm_jpos_des


    @property
    def ee_pos_w(self) -> torch.Tensor:
        """Current EE position in world frame, shape ``(num_envs, 3)``."""
        return self.robot.data.body_pose_w[:, self.ee_idx, :3]

    @property
    def ee_quat_w(self) -> torch.Tensor:
        """Current EE orientation (w,x,y,z) in world frame, shape ``(num_envs, 4)``."""
        return self.robot.data.body_pose_w[:, self.ee_idx, 3:]
