# Created by skywoodsz on 5/21/26.

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from isaaclab.envs import mdp

ACTION_TERM_NAMES = {
    "leg": "joint_leg",
    "arm": "joint_arm",
    "wheel": "joint_wheel",
}

# Default configuration
DEFAULT_ACTION_SPEC: dict[str, dict[str, Any]] = {
    "leg": {
        "mode": "position",
        "scale": 0.5,
        "clip": None,
    },
    "arm": {
        "mode": "position",
        "scale": 0.5,
        "clip": None,
    },
    "wheel": {
        "mode": "velocity",
        "scale": 5.0,
        "clip": None,
    },
}

# Custom configuration
ALLOWED_GROUPS = {"leg", "arm", "wheel"}
ALLOWED_MODES = {
    "leg": {"position", "velocity", "effort"},
    "arm": {"position", "velocity", "effort"},
    "wheel": {"position", "velocity", "effort"},
}
SCALE_LIMITS = {
    "leg": (1.0e-6, 100.0),
    "arm": (1.0e-6, 100.0),
    "wheel": (1.0e-6, 100.0),
}



def _merge_action_spec(user_spec: dict[str, Any] | None, source_name: str) -> dict[str, dict[str, Any]]:
    """Merge a partial participant action spec with the official defaults."""
    spec = deepcopy(DEFAULT_ACTION_SPEC)

    if user_spec is None:
        return spec

    if not isinstance(user_spec, dict):
        raise TypeError(f"{source_name} must be a dict or None.")

    for group_name, group_spec in user_spec.items():
        if group_name not in ALLOWED_GROUPS:
            raise ValueError(
                f"Unsupported action group '{group_name}'. "
                f"Allowed groups are {sorted(ALLOWED_GROUPS)}."
            )

        if not isinstance(group_spec, dict):
            raise TypeError(f"Action spec for group '{group_name}' must be a dict.")

        for key in group_spec.keys():
            if key not in {"mode", "scale", "clip"}:
                raise ValueError(
                    f"Unsupported key '{key}' in action spec for group '{group_name}'. "
                    f"Only 'mode', 'scale', and 'clip' are allowed."
                )

        spec[group_name].update(group_spec)

    return spec


def get_participant_action_spec(action_spec_source: Any) -> dict[str, dict[str, Any]]:
    """Read optional participant action spec from an object, dict, or JSON string.

    Supported inputs:
        - AlgSolution-like object with get_action_spec()
        - dict returned by get_action_spec()
        - JSON string serialized from that dict
        - None or JSON null, which uses defaults
    """
    if isinstance(action_spec_source, str):
        try:
            user_spec = json.loads(action_spec_source)
        except json.JSONDecodeError as exc:
            raise ValueError("Action spec JSON string is invalid.") from exc
        return _merge_action_spec(user_spec, "Action spec JSON")

    if isinstance(action_spec_source, dict) or action_spec_source is None:
        return _merge_action_spec(action_spec_source, "Action spec")

    if not hasattr(action_spec_source, "get_action_spec"):
        return _merge_action_spec(None, "Action spec")

    user_spec = action_spec_source.get_action_spec()
    return _merge_action_spec(user_spec, "AlgSolution.get_action_spec()")

def validate_action_spec(spec: dict[str, dict[str, Any]]) -> None:
    for group_name, group_spec in spec.items():
        if group_name not in ALLOWED_GROUPS:
            raise ValueError(
                f"Unsupported action group '{group_name}'. "
                f"Allowed groups are {sorted(ALLOWED_GROUPS)}."
            )

        mode = group_spec.get("mode")
        if mode not in ALLOWED_MODES[group_name]:
            raise ValueError(
                f"Unsupported action mode '{mode}' for group '{group_name}'. "
                f"Allowed modes are {sorted(ALLOWED_MODES[group_name])}."
            )

        scale = group_spec.get("scale")
        if not isinstance(scale, (int, float)):
            raise TypeError(f"'scale' for group '{group_name}' must be a number.")

        min_scale, max_scale = SCALE_LIMITS[group_name]
        if not (min_scale <= float(scale) <= max_scale):
            raise ValueError(
                f"'scale' for group '{group_name}' must be in "
                f"[{min_scale}, {max_scale}], got {scale}."
            )

        clip = group_spec.get("clip", None)
        if clip is not None:
            if not isinstance(clip, (list, tuple)) or len(clip) != 2:
                raise TypeError(f"'clip' for group '{group_name}' must be None or [min, max].")

            clip_min, clip_max = clip

            if not isinstance(clip_min, (int, float)) or not isinstance(clip_max, (int, float)):
                raise TypeError(f"'clip' values for group '{group_name}' must be numbers.")

            if float(clip_min) >= float(clip_max):
                raise ValueError(
                    f"'clip' lower bound must be smaller than upper bound for group '{group_name}'."
                )

def _get_action_metadata(action_cfg: Any, group_name: str) -> dict[str, Any]:
    """Read metadata from existing task/robot action cfg.

    This is how we inherit joint_names and joint order from the current task/robot.
    """
    joint_names = getattr(action_cfg, "joint_names", None)

    if joint_names is None:
        raise AttributeError(
            f"Action cfg for group '{group_name}' does not have 'joint_names'."
        )

    return {
        "asset_name": getattr(action_cfg, "asset_name", "robot"),
        "joint_names": joint_names,
        "preserve_order": getattr(action_cfg, "preserve_order", True),
        "use_default_offset": getattr(action_cfg, "use_default_offset", True),
    }

def _make_action_cfg(group_spec: dict[str, Any], metadata: dict[str, Any]) -> Any:
    mode = group_spec["mode"]
    scale = float(group_spec["scale"])
    clip = group_spec.get("clip", None)
    clip_cfg = {".*": tuple(clip)} if clip is not None else None

    common_kwargs = {
        "asset_name": metadata["asset_name"],
        "joint_names": metadata["joint_names"],
        "scale": scale,
        "clip": clip_cfg,
        "preserve_order": metadata["preserve_order"],
    }

    if mode == "position":
        return mdp.JointPositionActionCfg(
            use_default_offset=metadata["use_default_offset"],
            **common_kwargs,
        )

    if mode == "velocity":
        return mdp.JointVelocityActionCfg(
            **common_kwargs,
        )

    if mode == "effort":
        return mdp.JointEffortActionCfg(
            **common_kwargs,
        )

    raise ValueError(f"Unsupported action mode: {mode}")


def apply_safe_action_spec(env_cfg: Any, action_spec_source: Any, verbose: bool = True) -> Any:
    """Apply participant action spec safely.

    Participants cannot touch env_cfg.
    They can only define AlgSolution.get_action_spec(), or provide its JSON string.
    """
    spec = get_participant_action_spec(action_spec_source)
    validate_action_spec(spec)

    actions = env_cfg.actions

    for group_name, group_spec in spec.items():
        term_name = ACTION_TERM_NAMES[group_name]

        if not hasattr(actions, term_name):
            if verbose:
                print(f"[INFO] Action group '{group_name}' is not available. Skip.")
            continue

        old_action_cfg = getattr(actions, term_name)

        if old_action_cfg is None:
            if verbose:
                print(f"[INFO] Action term '{term_name}' is None. Skip.")
            continue

        metadata = _get_action_metadata(old_action_cfg, group_name)

        new_action_cfg = _make_action_cfg(
            group_spec=group_spec,
            metadata=metadata,
        )

        setattr(actions, term_name, new_action_cfg)

        if verbose:
            print(
                f"[INFO] Applied action spec for {group_name}: "
                f"term={term_name}, "
                f"mode={group_spec['mode']}, "
                f"scale={group_spec['scale']}, "
                f"clip={group_spec.get('clip', None)}, "
                f"joint_names={metadata['joint_names']}"
            )

    env_cfg.actions = actions
    return env_cfg
