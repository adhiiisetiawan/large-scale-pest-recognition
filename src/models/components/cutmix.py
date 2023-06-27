import torch
from torch.distributions import Beta, Uniform


def cutmix(minibatch, alpha):
    
    data, targets = minibatch
    indices = torch.randperm(data.size(0))
    data_shuffled = data[indices]
    target_shuffled = targets[indices]

    # Beta distribution sampling
    lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample()
    img_H, img_W = data.size(-2), data.size(-1)
    
    #uniform and random bounding box
    rx = Uniform(0, img_W).sample()
    ry = Uniform(0,img_H).sample()
    rw = img_W * torch.sqrt(1 - lam)
    rh = img_H * torch.sqrt(1 - lam)

    x1 = torch.round(torch.clamp(rx - rw // 2, 0, img_W))
    y1 = torch.round(torch.clamp(ry - rh // 2, 0, img_H))
    x2 = torch.round(torch.clamp(rx + rw // 2, 0, img_W))
    y2 = torch.round(torch.clamp(ry + rh // 2, 0, img_H))
     
    lam = 1 - ((x2 - x1)  * (y2 - y1) / (img_H * img_W))
    data[ : , : , int(y1) : int(y2), int(x1) : int(x2)] = data_shuffled[ : , : , int(y1) : int(y2), int(x1) : int(x2) ] 

    return data, targets, target_shuffled, lam