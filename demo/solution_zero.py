import torch

class AlgSolution:

    def __init__(self):
        pass

    def predicts(self, obs, current_score):
        proprio = obs['proprio']
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        action = [0 for _ in range(action_dim)]
        return {'action': action, 'giveup': False}
