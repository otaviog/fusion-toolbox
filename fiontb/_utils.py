import numpy as np
import torch


def ensure_torch(x, dtype=None):
    if isinstance(x, (np.ndarray, list, tuple)):
        x = torch.from_numpy(x)
    if dtype is not None:
        return x.type(dtype)
    return x


def empty_ensured_size(tensor, *sizes, dtype=torch.float, device=None):
    """Returns an empty allocated tensor if the input tensor is None or
    if its sizes are different than input.

    """

    if tensor is None or tensor.size() != sizes:
        return torch.empty(sizes, dtype=dtype, device=device)

    return tensor
