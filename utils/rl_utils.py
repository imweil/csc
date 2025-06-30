import torch
import torch


def compute_advantage(gamma, lmbda, td_delta):
    advantage = 0.0
    advantage_list = []
    for delta in torch.flip(td_delta, [0]):
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)

    advantage_tensor = torch.tensor(advantage_list, dtype=torch.float)
    return advantage_tensor.flip(0)
