"""Frame data types.
"""

import numpy as np
import torch

import fiontb.fiontblib as fiontblib
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
         int16.

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


class FramePointCloud:
    """Pre point cloud data.
    """

    def __init__(self, frame: Frame):
        info = frame.info

        self.depth_image = (frame.depth_image.astype(np.float64)*info.depth_scale +
                            info.depth_bias).astype(np.float32)

        self.depth_mask = frame.depth_image > 0
        self.fg_mask = self.depth_mask
        if frame.fg_mask is not None:
            self.fg_mask = np.logical_and(frame.fg_mask, self.depth_mask)

        xs, ys = np.meshgrid(np.arange(frame.depth_image.shape[1]),
                             np.arange(frame.depth_image.shape[0]))
        self.image_points = np.dstack(
            [xs, ys, self.depth_image]).astype(np.float32)

        if frame.rgb_image is not None:
            self.colors = frame.rgb_image

        self.kcam = info.kcam
        self.rt_cam = info.rt_cam
        self._points = None
        self._normals = None

        if frame.normal_image is not None:
            self._normals = frame.normal_image

    @property
    def points(self):
        """Points in the camera space property.

        Returns: (:obj:`numpy.ndarray`): [Nx3x1] array of points in the camera space.
        """

        if self._points is None:
            if self.kcam is None:
                raise RuntimeError("Frame doesn't have intrinsics camera")

            self._points = self.kcam.backproject(
                self.image_points.reshape(-1, 3)).reshape(self.image_points.shape)
        return self._points

    @property
    def normals(self):
        if self._normals is None:
            self._normals = fiontblib.calculate_depth_normals(
                torch.from_numpy(self.points),
                torch.from_numpy(self.depth_mask.astype(np.uint8))).numpy()

        return self._normals

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
