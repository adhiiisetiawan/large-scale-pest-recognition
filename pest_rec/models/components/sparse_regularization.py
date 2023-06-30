import torch
import torch.nn.functional as F

def sparse_loss(model, images):
    loss = 0
    values = images
    for module in model.children():
        values = F.relu6(module(values))
        loss += torch.mean(torch.abs(values))
    return loss