import numpy as np
import torch


def ensure_torch(x, dtype=None):
    if isinstance(x, (np.ndarray, list, tuple)):
        x = torch.from_numpy(x)
    if dtype is not None:
        return x.type(dtype)
    return x
