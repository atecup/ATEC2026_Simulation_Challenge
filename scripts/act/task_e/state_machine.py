"""Pick-place state machine and grasp-quaternion solver for Task E."""

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix

from .config import (
    STEPS, STATE_ORDER,
    CARRY_Z, PLACE_HEIGHT,
    RETRACT_POS_X, RETRACT_POS_Y,
    GRASP_Z_OFFSET,
    BASKET_CENTER_X, BASKET_CENTER_Y,
    DEFAULT_PLACE_QUAT_W,
)


def _build_grasp_matrix(long_axis: torch.Tensor, grip_z: torch.Tensor) -> torch.Tensor:
    """Build a right-handed gripper rotation matrix given the object's long axis.

    The Piper gripper jaw opens along its LOCAL Y axis, so local Y must be
    perpendicular to the object's long axis.

    Frame layout (columns of R_grip):
      col 0 (local X) = align_dir  ∥ long_axis
      col 1 (local Y) = jaw_dir    ⊥ long_axis  ← jaw opening direction
      col 2 (local Z) = grip_z     pointing down

    Right-hand check: col0 × col1 = align_dir × jaw_dir = grip_z ✓
    """
    jaw_dir   = torch.linalg.cross(long_axis, grip_z)   # ⊥ long_axis, in XY plane
    jaw_dir   = jaw_dir / jaw_dir.norm().clamp(min=1e-6)
    align_dir = torch.linalg.cross(jaw_dir, grip_z)     # ∥ long_axis
    align_dir = align_dir / align_dir.norm().clamp(min=1e-6)
    return torch.stack([align_dir, jaw_dir, grip_z], dim=1)   # (3, 3)


def compute_grasp_quat(obj_quat_w: torch.Tensor, device: str) -> torch.Tensor:
    """Compute a top-down grasp quaternion for the object.

    Finds the object axis most aligned with the world XY-plane (the long axis),
    builds a gripper frame where the jaw (local Y) is perpendicular to that axis,
    then picks the candidate orientation closest to the default top-down pose.

    Parameters
    ----------
    obj_quat_w : (4,) tensor, (w, x, y, z)
    device     : torch device string

    Returns
    -------
    grasp_quat : (4,) tensor, (w, x, y, z)
    """
    R_obj        = matrix_from_quat(obj_quat_w.unsqueeze(0)).squeeze(0)   # (3, 3)
    grip_z       = torch.tensor([0.0, 0.0, -1.0], device=device)
    default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=device)

    # Project each object column-axis onto XY plane; keep the most horizontal one(s)
    norms, axes_xy = [], []
    for col in range(3):
        ax = torch.tensor([R_obj[0, col].item(), R_obj[1, col].item(), 0.0], device=device)
        norms.append(ax.norm().item())
        axes_xy.append(ax)

    best_norm  = max(norms)
    candidates = [
        axes_xy[c] / max(norms[c], 1e-6)
        for c in range(3)
        if norms[c] >= best_norm - 1e-3
    ]

    # Among candidates, pick the one whose grasp frame is closest to the default orientation
    best_cos  = -2.0
    long_axis = candidates[0]
    for cand in candidates:
        q_cand  = quat_from_matrix(_build_grasp_matrix(cand, grip_z).unsqueeze(0)).squeeze(0)
        cos_sim = torch.abs((q_cand * default_quat).sum()).item()
        if cos_sim > best_cos:
            best_cos  = cos_sim
            long_axis = cand

    R_grip = _build_grasp_matrix(long_axis, grip_z)                  # (3, 3)
    return quat_from_matrix(R_grip.unsqueeze(0)).squeeze(0)           # (4,) w,x,y,z


class PickPlaceStateMachine:
    """Finite state machine that sequences pick-and-place for multiple objects.

    States (in order): INIT → PRE_GRASP → REACH → CLOSE → LIFT →
                       TRANSPORT → PLACE → OPEN → RETRACT → (next object or done)
    """

    def __init__(self, object_indices: list[int], device: str):
        self._obj_indices = object_indices
        self._device      = device
        self._grasp_quat_cache: dict[int, torch.Tensor] = {}
        self.reset()

    def reset(self) -> None:
        self._ptr             = 0
        self._state_idx       = 0
        self._count           = 0
        self.done             = False
        self._cached_obj_pos: torch.Tensor | None = None
        self._grasp_quat_cache.clear()

    def set_grasp_quat(self, obj_idx: int, obj_quat_w: torch.Tensor) -> None:
        """Pre-compute and cache the grasp quaternion for one object."""
        self._grasp_quat_cache[obj_idx] = compute_grasp_quat(obj_quat_w, self._device)

    def tick(self, obj_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, str]:
        """Advance the state machine by one step.

        Parameters
        ----------
        obj_pos : (3,) tensor — current object position in world frame

        Returns
        -------
        ee_pos_des  : (3,) target EE position
        ee_quat_des : (4,) target EE orientation (w,x,y,z)
        gripper_cmd : "open" | "close"
        """
        s = self.state
        d = self._device

        # Freeze object position at start of PRE_GRASP to avoid drift during descent
        if s == "PRE_GRASP" and self._count == 0:
            self._cached_obj_pos = obj_pos.clone()
        if s in ("REACH", "CLOSE") and self._cached_obj_pos is not None:
            obj_pos = self._cached_obj_pos

        ee_pos, gripper = self._get_target_pos_gripper(s, obj_pos, d)
        ee_quat         = self._get_target_quat(s, d)

        self._count += 1
        if self._count >= STEPS[s]:
            self._count = 0
            if s == "RETRACT":
                self._ptr          += 1
                self._cached_obj_pos = None
                if self._ptr >= len(self._obj_indices):
                    self.done = True
                    return ee_pos, ee_quat, gripper
                self._state_idx = STATE_ORDER.index("PRE_GRASP")
            else:
                self._state_idx += 1

        return ee_pos, ee_quat, gripper

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> str:
        return STATE_ORDER[self._state_idx]

    @property
    def current_object_key(self) -> str:
        return f"object_{self._obj_indices[self._ptr]}"

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_target_pos_gripper(
        self, s: str, obj_pos: torch.Tensor, d: str
    ) -> tuple[torch.Tensor, str]:
        if s == "INIT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        elif s == "PRE_GRASP":
            p = obj_pos.clone(); p[2] = CARRY_Z
            return p, "open"
        elif s == "REACH":
            p = obj_pos.clone(); p[2] += GRASP_Z_OFFSET
            return p, "open"
        elif s == "CLOSE":
            p = obj_pos.clone(); p[2] += GRASP_Z_OFFSET
            return p, "close"
        elif s == "LIFT":
            p = obj_pos.clone(); p[2] = CARRY_Z
            return p, "close"
        elif s == "TRANSPORT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "close"
        elif s == "PLACE":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "close"
        elif s == "OPEN":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, PLACE_HEIGHT], device=d), "open"
        elif s == "LIFT_RETRACT":
            return torch.tensor([BASKET_CENTER_X, BASKET_CENTER_Y, CARRY_Z], device=d), "open"
        elif s == "RETRACT":
            return torch.tensor([RETRACT_POS_X, RETRACT_POS_Y, CARRY_Z], device=d), "open"
        else:
            raise ValueError(f"Unknown state: {s}")

    def _get_target_quat(self, s: str, d: str) -> torch.Tensor:
        default_quat = torch.tensor(DEFAULT_PLACE_QUAT_W, dtype=torch.float32, device=d)
        if s in ("REACH", "CLOSE", "LIFT"):
            cur_idx = self._obj_indices[min(self._ptr, len(self._obj_indices) - 1)]
            return self._grasp_quat_cache.get(cur_idx, default_quat)
        return default_quat
