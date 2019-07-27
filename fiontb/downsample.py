from enum import Enum

import torch
import torch.nn.functional
import cv2

from fiontb._cfiontb import (downsample_xyz as _downsample_xyz,
                             DownsampleXYZMethod)


def downsample_xyz(xyz_image, mask, scale, dst=None, normalize=False,
                   method=DownsampleXYZMethod.Nearest):
    if dst is None:
        size0 = int(xyz_image.size(0)*scale)
        size1 = int(xyz_image.size(1)*scale)
        dst = torch.empty(size0, size1, 3, dtype=xyz_image.dtype,
                          device=xyz_image.device)

    _downsample_xyz(xyz_image, mask, scale, dst,
                    normalize, method)

    return dst


class DownsampleMethod(Enum):
    Nearest = 0


def downsample(image, scale, method=DownsampleMethod.Nearest):
    method_to_cv = {DownsampleMethod.Nearest: cv2.INTER_NEAREST}
    device = image.device

    height = int(image.size(0)*scale)
    width = int(image.size(1)*scale)

    image = cv2.resize(image.cpu().numpy(), (width, height),
                       interpolation=method_to_cv[method])
    image = torch.from_numpy(image).to(device)

    return image
