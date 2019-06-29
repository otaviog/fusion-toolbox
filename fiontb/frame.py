c"""Frame data types.
"""

import numpy as np
import torch

from fiontb._cfiontb import estimate_normals as _estimate_normals
from fiontb._cfiontb import EstimateNormalsMethod

from fiontb._utils import ensure_torch
from .camera import KCamera, RTCamera
from .pointcloud import PointCloud


class FrameInfo:
    """Frame information data.

    Attributes:

        kcam (:obj:`fiontb.camera.KCamera`): Intrinsic camera information.
        depth_scale (float): Scaling of depth values.
        depth_bias (float): Constant added to depth values.
        depth_max (float): Max depth value.
        rt_cam (:obj:`fiontb.camera.RTCamera`): Camera to world matrix.
        timestamps (float or int): Sensor's timestamp
    """

    def __init__(self, kcam=None, depth_scale=1.0, depth_bias=0.0, depth_max=4500.0,
                 timestamp=None, rgb_kcam=None, rt_cam=None):
        self.kcam = kcam
        self.depth_scale = depth_scale
        self.depth_bias = depth_bias
        self.depth_max = depth_max

        self.rgb_kcam = rgb_kcam
        self.rt_cam = rt_cam
        self.timestamp = timestamp

    @classmethod
    def from_json(cls, json):
        """Creates a frame info from its json dict representation
        """

        kcam = json.get('kcam', None)
        if kcam is not None:
            kcam = KCamera.from_json(kcam)

        rgb_kcam = json.get('rgb_kcam', None)
        if rgb_kcam is not None:
            rgb_kcam = KCamera.from_json(rgb_kcam)

        rt_cam = json.get('rt_cam', None)
        if rt_cam is not None:
            rt_cam = RTCamera.from_json(rt_cam)

        return cls(kcam, json.get('depth_scale', 1.0),
                   json.get('depth_bias', 0.0),
                   json.get('depth_max', 4500.0),
                   json.get('timestamp', None),
                   rgb_kcam, rt_cam)

    def to_json(self):
        """Converts the frame info to its json dict representation.

        Returns: (dict): Dict ready for json dump.
        """
        json = {}
        for name, value in vars(self).items():
            if value is None:
                continue
            if hasattr(value, 'to_json'):
                value = value.to_json()

            json[name] = value

        return json

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(self)


class Frame:
    """Frame data:

    Attributes:

        info (:obj:`FrameInfo`): Frame information.

        depth_image (:obj:`numpy.ndarray`): Depth image [WxH] float or
         int32.

        rgb_image (:obj:`numpy.ndarray`, optional): RGB image [WxHx3]
         uint8.

        fg_mask (:obj:`ndarray.ndarray`, optional): Mask image [WxH]
         bool or uint8.
    """

    def __init__(self, info: FrameInfo, depth_image, rgb_image=None, fg_mask=None, normal_image=None):
        self.info = info
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.fg_mask = fg_mask
        self.normal_image = normal_image


class _DepthImagePointCloud:
    def __init__(self, depth_image, finfo):
        depth_image = ensure_torch(depth_image)
        self.depth_image = (depth_image.float()*finfo.depth_scale +
                            finfo.depth_bias)

        self.depth_mask = depth_image > 0
        device = self.depth_image.device
    
        self.kcam = finfo.kcam

        ys, xs = torch.meshgrid(torch.arange(self.depth_image.size(0), dtype=torch.float),
                                torch.arange(self.depth_image.size(1), dtype=torch.float))

        self.image_points = torch.stack(
            [xs.to(device), ys.to(device), self.depth_image], 2)

        self._points = None

    @property
    def points(self):
        """Points in the camera space property.

        Returns: (:obj:`numpy.ndarray`): [WxHx3] array of points in the camera space.
        """

        if self._points is None:
            if self.kcam is None:
                raise RuntimeError("Frame doesn't have intrinsics camera")

            self._points = self.kcam.backproject(
                self.image_points.reshape(-1, 3)).reshape(self.image_points.shape)
        return self._points


class FramePointCloud(_DepthImagePointCloud):
    """A point cloud ordered by image positions.
    """

    def __init__(self, frame: Frame):
        super(FramePointCloud, self).__init__(frame.depth_image, frame.info)

        self.fg_mask = self.depth_mask
        if frame.fg_mask is not None:
            self.fg_mask = torch.logical_and(
                torch.from_numpy(frame.fg_mask).byte(), self.depth_mask)

        if frame.rgb_image is not None:
            self.colors = torch.from_numpy(frame.rgb_image)

        self.rt_cam = frame.info.rt_cam

        self._normals = None
        if frame.normal_image is not None:
            self._normals = frame.normal_image

    @property
    def normals(self):
        if self._normals is None:
            self._normals = _calculate_depth_normals(
                self.points, self.depth_mask)

        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    def unordered_point_cloud(self, world_space=True):
        mask = self.fg_mask.flatten()

        normals = self.normals.reshape(-1, 3)
        normals = normals[mask]

        pcl = PointCloud(self.points.reshape(-1, 3)[mask],
                         self.colors.reshape(-1, 3)[mask],
                         normals)

        if world_space and self.rt_cam is not None:
            pcl.transform(self.rt_cam.cam_to_world)

        return pcl


def estimate_normals(depth_image, frame_info, mask,
                     method=EstimateNormalsMethod.CentralDifferences,
                     out_tensor=None):
    pcl = _DepthImagePointCloud(depth_image, frame_info)
    xyz_img = pcl.points

    if out_tensor is None:
        out_tensor = torch.empty(xyz_img.size(0), xyz_img.size(1), 3, dtype=xyz_img.dtype,
                                 device=xyz_img.device)

    _estimate_normals(xyz_img, ensure_torch(mask, dtype=torch.uint8),
                      out_tensor, method)

    return out_tensor
