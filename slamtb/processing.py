"""Common filtering functions used by 3d reconstruction on frames.
"""

from enum import Enum

import torch
import torch.nn.functional
from torchvision.transforms.functional import to_tensor
from kornia.filters import GaussianBlur2d
import cv2

from slamtb._cslamtb import (Processing as _Processing, EstimateNormalsMethod,
                             DownsampleXYZMethod)
from slamtb._utils import ensure_torch, empty_ensured_size, depth_image_to_uvz


def bilateral_depth_filter(depth, mask=None, out_tensor=None, filter_width=6,
                           sigma_color=29.9999880000072,
                           sigma_space=4.50000000225,
                           depth_scale=1.0):
    """Apply bilateral filter on depth image.

    Default argument values are the same ones used by ElasticFusion.

    Args:

        depth (torch.Tensor or numpy.ndarray): Input depth image. May
         have any dtype.

        out_tensor (torch.Tensor, optional): Optional output tensor.

        filter_width (int): Bilateral filter size. Default is 6.

        sigma_color (float):

        sigma_space (float):

        depth_scale (float). Scaling factor for depth values before
         filtering. Default is no scaling.

    """

    depth = ensure_torch(depth)
    if mask is None:
        mask = depth > 0
    else:
        mask = ensure_torch(mask, dtype=torch.bool)

    out_tensor = empty_ensured_size(out_tensor, depth.size(0), depth.size(1),
                                    dtype=depth.dtype, device=depth.device)

    _Processing.bilateral_depth_filter(depth, mask, out_tensor,
                                       filter_width, sigma_color, sigma_space, depth_scale)

    return out_tensor


class BilateralDepthFilter:
    def __init__(self, filter_width=6, sigma_color=29.9999880000072,
                 sigma_space=4.50000000225, depth_scale=1.0):
        self.filter_width = filter_width
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.depth_scale = depth_scale

        self._out_tensor = None

    def __call__(self, depth, mask):
        self._out_tensor = empty_ensured_size(self._out_tensor,
                                              depth.size(0), depth.size(1),
                                              device=depth.device,
                                              dtype=depth.dtype)
        _Processing.bilateral_depth_filter(depth, mask, self._out_tensor,
                                           self.filter_width, self.sigma_color,
                                           self.sigma_space, self.depth_scale)
        return self._out_tensor


#############################
# Normals
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
# Downsample
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


def downsample_features(feature_image, scale, dst=None):
    """Downsam
    """
    dst = empty_ensured_size(dst,
                             feature_image.size(0),
                             int(feature_image.size(1)*scale),
                             int(feature_image.size(2)*scale),
                             dtype=feature_image.dtype,
                             device=feature_image.device)
    _Processing.downsample_features(feature_image, scale, dst)
    return dst


def feature_pyramid(feature_map, scales):
    """Build a gaussian pyramid for featuremaps.

    """

    pyramid = []

    blur = GaussianBlur2d((3, 3), (0.849, 0.849))

    for scale in scales:
        if scale < 1.0:
            feature_map = downsample_features(
                blur(feature_map.unsqueeze(0)).squeeze(0),
                # feature_map,
                scale)
        pyramid.append(feature_map)

    pyramid.reverse()
    return pyramid

################################
# Erode
################################


def erode_mask(in_mask):
    out_mask = torch.empty(in_mask.size(), dtype=torch.bool,
                           device=in_mask.device)

    _Processing.erode_mask(in_mask, out_mask)
    return out_mask

################################
# Color
################################


class ColorSpace(Enum):
    """Specify color spaces for preprocessing functions.
    """
    RGB = 0
    GRAY = 1
    INTENSITY = 1
    LAB = 2
    HSV = 3


def to_color_feature(rgb_image, color_space=ColorSpace.RGB, blur=False):
    """Preprocess and reshape a rgb image into a feature format.
    """
    features = rgb_image

    if blur:
        features = cv2.blur(features, (5, 5))

    stb2cv2 = {
        ColorSpace.LAB: cv2.COLOR_RGB2LAB,
        ColorSpace.HSV: cv2.COLOR_RGB2HSV,
        ColorSpace.GRAY: cv2.COLOR_RGB2GRAY,
        ColorSpace.INTENSITY: cv2.COLOR_RGB2GRAY
    }

    if color_space != ColorSpace.RGB:
        features = cv2.cvtColor(features, stb2cv2[color_space])

    return to_tensor(features)
