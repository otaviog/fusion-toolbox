from enum import Enum

import torch
import torch.nn.functional
import cv2

from fiontb._cfiontb import (Downsample as _Downsample,
                             DownsampleXYZMethod)
from fiontb._utils import empty_ensured_size


def downsample_xyz(xyz_image, mask, scale, dst=None, normalize=False,
                   method=DownsampleXYZMethod.Nearest):
    dst = empty_ensured_size(dst, int(xyz_image.size(0)*scale),
                             int(xyz_image.size(1)*scale),
                             xyz_image.size(2),
                             dtype=xyz_image.dtype,
                             device=xyz_image.device)

    _Downsample.downsample_xyz(xyz_image, mask, scale, dst,
                               normalize, method)

    return dst


def downsample_mask(mask, scale, dst=None):
    dst = empty_ensured_size(dst, int(mask.size(0)*scale),
                             int(mask.size(1)*scale),
                             dtype=mask.dtype,
                             device=mask.device)
    _Downsample.downsample_mask(mask, scale, dst)

    return dst


class DownsampleMethod(Enum):
    Nearest = 0


def downsample(image, scale, method=DownsampleMethod.Nearest):
    if False:
        method_to_cv = {DownsampleMethod.Nearest: cv2.INTER_NEAREST}
        device = image.device

        height = int(image.size(0)*scale)
        width = int(image.size(1)*scale)

        image = cv2.resize(image.cpu().numpy(), (width, height),
                           interpolation=method_to_cv[method])
        image = torch.from_numpy(image).to(device)

        return image
    else:
        method_to_torch = {DownsampleMethod.Nearest: 'nearest'}
        return torch.nn.functional.interpolate(
            image.unsqueeze(0), scale_factor=scale,
            mode=method_to_torch[method]).squeeze()
