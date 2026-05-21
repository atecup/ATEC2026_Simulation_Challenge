import torch
from typing import Any

class AlgSolution:

    def __init__(self):
        pass

    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return {
        "leg": {
            "mode": "position",
            "scale": 1.0,
            "clip": [-10.0, 10.0],
        },
        "wheel": {
            "mode": "velocity",
            "scale": 2.0,
            "clip": [-11.0, 11.0],
        },
        "arm": {
            "mode": "effort",
            "scale": 3.0,
            "clip": [-12.0, 12.0],
        },
    }

    def predicts(self, obs, current_score):
        proprio = obs['proprio']
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0 for _ in range(action_dim)]
        return {'action': action, 'giveup': False}
