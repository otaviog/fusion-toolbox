"""Common filtering of 3d reconstruction for frames.
"""

from enum import Enum


import torch
import torch.nn.functional
import cv2

from fiontb._cfiontb import (Processing as _Processing, EstimateNormalsMethod,
                             DownsampleXYZMethod)
from fiontb._utils import ensure_torch, empty_ensured_size


def bilateral_depth_filter(depth, mask, out_tensor=None, filter_width=6,
                           sigma_d=4.50000000225,
                           sigma_r=29.9999880000072,
                           depth_scale=1.0):
    depth = ensure_torch(depth)
    mask = ensure_torch(mask, dtype=torch.bool)
    out_tensor = empty_ensured_size(out_tensor, depth.size(0), depth.size(1),
                                    dtype=depth.dtype, device=depth.device)

    _Processing.bilateral_depth_filter(depth, mask, out_tensor,
                                       filter_width, sigma_d, sigma_r, depth_scale)

    return out_tensor


class BilateralDepthFilter:
    def __init__(self, filter_width=6, sigma_d=4.50000000225,
                 sigma_r=29.9999880000072, depth_scale=1.0):
        self.filter_width = filter_width
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.depth_scale = depth_scale

        self._out_tensor = None

    def __call__(self, depth, mask):
        self._out_tensor = empty_ensured_size(self._out_tensor,
                                              depth.size(0), depth.size(1),
                                              device=depth.device,
                                              dtype=depth.dtype)
        _Processing.bilateral_depth_filter(depth, mask, self._out_tensor,
                                           self.filter_width, self.sigma_d, self.sigma_r,
                                           self.depth_scale)
        return self._out_tensor


#############################
# Normals                   #
#############################

def estimate_normals(depth_image, frame_info, mask,
                     method=EstimateNormalsMethod.CentralDifferences,
                     out_tensor=None):
    image_points = depth_image_to_uvz(ensure_torch(depth_image), frame_info)
    xyz_img = frame_info.kcam.backproject(
        image_points.reshape(-1, 3)).reshape(image_points.shape)

    if out_tensor is None:
        out_tensor = torch.empty(xyz_img.size(0), xyz_img.size(1), 3, dtype=xyz_img.dtype,
                                 device=xyz_img.device)

    _Processing.estimate_normals(xyz_img, ensure_torch(mask, dtype=torch.bool),
                                 out_tensor, method)

    return out_tensor

#############################
# Downsample                #
#############################


def downsample_xyz(xyz_image, mask, scale, dst=None, normalize=False,
                   method=DownsampleXYZMethod.Nearest):
    dst = empty_ensured_size(dst, int(xyz_image.size(0)*scale),
                             int(xyz_image.size(1)*scale),
                             xyz_image.size(2),
                             dtype=xyz_image.dtype,
                             device=xyz_image.device)

    _Processing.downsample_xyz(xyz_image, mask, scale, dst,
                               normalize, method)

    return dst


def downsample_mask(mask, scale, dst=None):
    dst = empty_ensured_size(dst, int(mask.size(0)*scale),
                             int(mask.size(1)*scale),
                             dtype=mask.dtype,
                             device=mask.device)
    _Processing.downsample_mask(mask, scale, dst)

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
