import torch
import numpy as np

def cutmix(data, targets, alpha=1.0):
    """
    Applies CutMix augmentation to the input data and targets.

    CutMix is a data augmentation technique that combines samples from different inputs by randomly
    selecting a bounding box in one sample and replacing the corresponding region in another sample.
    The targets are also mixed accordingly.

    Args:
        data (torch.Tensor): The input data tensor of shape (batch_size, channels, height, width).
        targets (torch.Tensor): The target tensor of shape (batch_size,) containing the class labels.
        alpha (float, optional): The hyperparameter controlling the strength of the CutMix augmentation.
            Higher values of alpha result in stronger augmentation. Default is 1.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]: A tuple containing the augmented data,
        original targets, shuffled targets, and the lambda value used for mixing.

    """
    
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = data.size(0)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)

    data[:, :, bbx1:bbx2, bby1:bby2] = shuffled_data[:, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))

    return data, targets, shuffled_targets, lam

def rand_bbox(size, lam):
    """
    Generates random bounding box coordinates based on the size of the input tensor and the lambda value.

    Args:
        size (Tuple[int, int, int, int]): The size of the input tensor (batch_size, channels, height, width).
        lam (float): The lambda value used for determining the size of the bounding box.

    Returns:
        Tuple[int, int, int, int]: A tuple containing the bounding box coordinates (bbx1, bby1, bbx2, bby2).

    """
    
    width = size[2]
    height = size[3]
    cut_ratio = np.sqrt(1. - lam)
    cut_w = (width * cut_ratio).astype(np.int32)
    cut_h = (height * cut_ratio).astype(np.int32)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    bbx1 = np.clip(cx - cut_w // 2, 0, width)
    bby1 = np.clip(cy - cut_h // 2, 0, height)
    bbx2 = np.clip(cx + cut_w // 2, 0, width)
    bby2 = np.clip(cy + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2
