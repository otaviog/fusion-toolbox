import contextlib
import cProfile

import numpy as np
import torch

def depth_image_to_uvz(depth_image, finfo):
    """Converts an depth image to a meshgrid of u (columns), v (rows) an z
     coordinates.

    Args:

        depth_image (:obj:`torch.Tensor`): [WxH] depth image.

        finfo (:obj:`FrameInfo`): The source frame description.

    Returns: (:obj:`torch.Tensor`): [WxHx3] the depth image with the u
     and v pixel coordinates.

    """

    depth_image = (depth_image.float()*finfo.depth_scale +
                   finfo.depth_bias)
    device = depth_image.device
    dtype = depth_image.dtype
    ys, xs = torch.meshgrid(torch.arange(depth_image.size(0), dtype=dtype),
                            torch.arange(depth_image.size(1), dtype=dtype))

    image_points = torch.stack(
        [xs.to(device), ys.to(device), depth_image], 2)

    return image_points

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

@contextlib.contextmanager
def profile(output_file, really=True):
    if really:
        prof = cProfile.Profile()
        prof.enable()
        yield
        prof.disable()

        prof.dump_stats(str(output_file))
    else:
        yield

        
