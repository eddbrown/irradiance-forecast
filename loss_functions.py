import torch

def multiplicative_l2_loss(predicted, real):
    return torch.mean(torch.prod((1 + torch.pow((predicted - real),2)),dim=1))

def weighted_mse_loss(predicted, target):
    batch_size, bands = predicted.shape
    weight = torch.ones(predicted.shape).to(predicted.device)
    for b in range(bands):
        weight[:,b] = bands - b
    return torch.mean(weight * (predicted - target) ** 2)

def heavy_weighted(predicted, target):
    batch_size, bands = predicted.shape
    weight = torch.ones(predicted.shape).to(predicted.device)
    for b in range(bands):
        if b == 0:
            weight[:,b] = 100
        elif b == 1:
            weight[:,b] = 100
        else:
            weight[:,b] = 1
    return torch.mean(weight * (predicted - target) ** 2)