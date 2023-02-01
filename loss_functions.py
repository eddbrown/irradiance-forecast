import torch

def multiplicative_l2_loss(predicted, real):
    return torch.mean(torch.prod((1 + torch.pow((predicted - real),2)),dim=1))

def weighted_mse_loss(predicted, target):
    batch_size, bands = predicted.shape
    weight = torch.ones(predicted.shape).to(predicted.device)
    for b in range(bands):
        weight[:,b] = bands - b
    return torch.mean(weight * (predicted - target) ** 2)