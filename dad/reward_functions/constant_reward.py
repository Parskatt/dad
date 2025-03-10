import torch
from typing import Optional


class ConstantReward:
    def __init__(self, *, th: float, eps: Optional[float] = 0.01):
        self.th = th
        self.eps = eps

    def __call__(self, distances: torch.Tensor):
        B, K = distances.shape
        good = distances.detach() < self.th
        pos_reward = good.float() / (good.float().mean(dim=1, keepdim=True) + self.eps)
        neg_reward = 0
        reward = pos_reward * good + neg_reward * good.logical_not()
        return reward
