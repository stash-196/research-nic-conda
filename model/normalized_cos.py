import numpy as np
import torch


def normalized_cos(x):
    scale = torch.linalg.norm(torch.min(x, dim=0)[0] - torch.max(x, dim=0)[0])
    res = torch.zeros(len(x), 1)
    avg = torch.sum(x, dim=0) / len(x)
    rs = [torch.linalg.norm(point - avg) for point in x]
    max_r = max(rs)
    for i, r in enumerate(rs):
        res[i, 0] = torch.cos(r / max_r * 2 * np.pi) + 1
    return res / 2
