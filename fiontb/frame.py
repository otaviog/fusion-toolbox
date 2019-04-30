"""Frame data types.
"""

import numpy as np
import cv2
import torch

from .camera import KCamera, RTCamera
from .datatypes import PointCloud


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

    def __init__(self, info: FrameInfo, depth_image, rgb_image=None, fg_mask=None):
        self.info = info
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.fg_mask = fg_mask


class FramePoints:
    """Pre point cloud data.
    """

    def __init__(self, frame: Frame):
        info = frame.info
        self.depth_image = (frame.depth_image*info.depth_scale +
                            info.depth_bias).astype(np.float32)

        depth_mask = frame.depth_image > 0
        if frame.fg_mask is not None:
            self.fg_mask = np.logical_and(frame.fg_mask, depth_mask)
        else:
            self.fg_mask = depth_mask

        xs, ys = np.meshgrid(np.arange(frame.depth_image.shape[1]),
                             np.arange(frame.depth_image.shape[0]))
        self.xyz_image = np.dstack(
            [xs, ys, self.depth_image]).astype(np.float32)

        self.points = self.xyz_image.reshape(-1, 3, 1)
        self.points = self.points[self.fg_mask.flatten()]

        if frame.rgb_image is not None:
            self.colors = frame.rgb_image.reshape(-1, 3)
            self.colors = self.colors[self.fg_mask.flatten()]

        self.kcam = info.kcam
        self._camera_points = None
        self._camera_xyz = None

    @property
    def camera_points(self):
        """Points in the camera space property.

        Returns: (:obj:`numpy.ndarray`): [Nx3x1] array of points in the camera space.
        """

        if self.kcam is None:
            raise RuntimeError("Frame doesn't have intrinsics camera")

        if self._camera_points is None:
            self._camera_points = self.kcam.backproject(self.points)
        return self._camera_points

    @property
    def camera_xyz_image(self):
        if self.kcam is None:
            raise RuntimeError("Frame doesn't have intrinsics camera")

        if self._camera_xyz is None:
            self._camera_xyz = self.kcam.backproject(
                self.xyz_image.reshape(-1, 3, 1))
            self._camera_xyz = self._camera_xyz.reshape(self.xyz_image.shape)

        return self._camera_xyz


def _compute_normals0(depth_img):
    zdy, zdx = np.gradient(depth_img)
    normals = np.dstack((-zdx, -zdy, np.ones_like(depth_img)))
    norm = np.linalg.norm(normals, axis=2)
    normals /= norm.reshape(norm.shape[0], norm.shape[1], 1)

    return normals


def compute_normals(depth_img):
    """Compute normals from a depth image.

    Args:

        depth_img (:obj:`numpy.ndarray`): Depth image [HxW].
    """
    depth_img = depth_img.astype(np.float32)
    filter1 = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]]) / 8
    filter2 = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]]) / 8

    filter1 = np.flip(filter1, 0)
    filter2 = np.flip(filter2, 1)

    img_x = cv2.filter2D(depth_img, -1, filter1, None,
                         (-1, -1), 0, cv2.BORDER_CONSTANT)
    img_y = cv2.filter2D(depth_img, -1, filter2, None,
                         (-1, -1), 0, cv2.BORDER_CONSTANT)

    img_x *= -1
    img_y *= -1

    norms = np.sqrt(img_x*img_x + img_y*img_y + 1)

    img_z = 1.0 / norms
    img_x *= img_z
    img_y *= img_z

    normals = np.dstack([img_x, img_y, img_z])
    return normals


def frame_to_pointcloud(frame):
    points = FramePoints(frame)
    import tenviz
    #normals = compute_normals(points.depth_image)
    # normals = tenviz.calculate_depth_normals(torch.from_numpy(points.depth_image)).numpy()

    normals = tenviz.calculate_depth_normals2(
        torch.from_numpy(points.camera_xyz_image),
        torch.from_numpy(points.fg_mask.astype(np.uint8))).numpy()
    normals = normals.reshape(-1, 3)
    normals = normals[points.fg_mask.flatten()]

    live_pcl = PointCloud(points.camera_points,
                          points.colors, normals)

    return live_pcl
