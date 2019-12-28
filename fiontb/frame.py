"""Frame related data types.
"""

import torch
from torch.nn.functional import interpolate

from fiontb.processing import (
    downsample_xyz, downsample_mask, DownsampleXYZMethod, erode_mask)
from fiontb._cfiontb import (Processing as _Processing, EstimateNormalsMethod)
from fiontb._utils import ensure_torch, depth_image_to_uvz

from .camera import KCamera, RTCamera
from .pointcloud import PointCloud


class FrameInfo:
    """A frame description. Holds information about camera parameters,
    depth scaling, and time.

    Attributes:

        kcam (:obj:`fiontb.camera.KCamera`): The intrinsic camera.

        depth_scale (float): Specifies the amount that depth values should be multiplied.

        depth_bias (float): Constant added to depth values.

        depth_max (float): Sensor's maximum depth value.

        rt_cam (:obj:`fiontb.camera.RTCamera`): The extrinsic camera parameters.

        timestamp (float or int): The frame timestamp.

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
        """Creates a frame info from FTB json dict representation
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
        """Converts the frame info to its json dictionary representation.

        Returns: (dict): Dictionary ready for json dump.
        """
        json = {}
        for name, value in vars(self).items():
            if value is None:
                continue
            if hasattr(value, 'to_json'):
                value = value.to_json()

            json[name] = value

        return json

    def clone(self):
        """Creates a copy of this frame info.
        """

        return FrameInfo(
            self.kcam.clone(), self.depth_scale, self.depth_bias,
            self.depth_max, self.timestamp,
            None if self.rgb_kcam is None else self.rgb_kcam.clone(),
            None if self.rt_cam is None else self.rt_cam.clone())

    def __str__(self):
        return str(vars(self))

    def __repr__(self):
        return str(self)


class Frame:
    """A RGBD frame, either outputed by a sensor or dataset.

    Attributes:

        info (:obj:`FrameInfo`): The information.

        depth_image (:obj:`numpy.ndarray`): Depth image [WxH] float or
         int32.

        rgb_image (:obj:`numpy.ndarray`, optional): RGB image [WxHx3]
         uint8.

        fg_mask (:obj:`ndarray.ndarray`, optional): Foreground mask image [WxH]
         bool or uint8.

    """

    def __init__(self, info: FrameInfo, depth_image, rgb_image=None,
                 fg_mask=None, normal_image=None):
        self.info = info
        self.info.kcam.image_size = (depth_image.shape[1],
                                     depth_image.shape[0])
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.fg_mask = fg_mask
        self.normal_image = normal_image

    def clone(self):
        return Frame(self.info.clone(), self.depth_image.copy(),
                     None if self.rgb_image is None else self.rgb_image.copy(),
                     None if self.fg_mask is None else self.fg_mask.copy(),
                     None if self.normal_image is None else self.normal_image.copy())

    def clone_shallow(self):
        return Frame(self.info, self.depth_image,
                     self.rgb_image, self.fg_mask, self.normal_image)


class _DepthImagePointCloud:
    def __init__(self, depth_image, finfo):
        depth_image = ensure_torch(depth_image)
        self.depth_image = (depth_image.float()*finfo.depth_scale +
                            finfo.depth_bias)

        self.mask = depth_image > 0
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


class FramePointCloud:
    """A framed point cloud: point, normal or color can be retrivied by
     pixel coordinates.

    Attributes:


    """

    def __init__(self, image_points, mask, kcam, rt_cam=None, points=None,
                 normals=None, colors=None):
        self.image_points = image_points
        self.mask = mask
        self.kcam = kcam
        self.rt_cam = rt_cam
        self._points = points
        self._normals = normals
        self.colors = colors

    @classmethod
    def from_frame(cls, frame: Frame):
        depth_image = ensure_torch(frame.depth_image)

        image_points = depth_image_to_uvz(depth_image, frame.info)

        mask = depth_image > 0
        mask = erode_mask(mask)
        if frame.fg_mask is not None:
            mask = torch.logical_and(
                torch.from_numpy(frame.fg_mask), mask)

        colors = None
        if frame.rgb_image is not None:
            colors = torch.from_numpy(frame.rgb_image)

        normals = None
        if frame.normal_image is not None:
            normals = ensure_torch(frame.normal_image)

        return FramePointCloud(image_points, mask, frame.info.kcam, frame.info.rt_cam,
                               normals=normals, colors=colors)

    @property
    def points(self):
        """Points in the camera space.

        Returns: (:obj:`numpy.ndarray`): [WxHx3] array of points in the camera space.
        """

        if self._points is None:
            if self.kcam is None:
                raise RuntimeError("Frame doesn't have intrinsics camera")

            self._points = self.kcam.backproject(
                self.image_points.reshape(-1, 3)).reshape(self.image_points.shape)
        return self._points

    @property
    def normals(self):
        """Normals.

        Returns: (:obj:`numpy.ndarray`): [WxHx3] array of normals in the camera space.
        """

        if self._normals is None:
            self._normals = torch.empty(self.points.size(0), self.points.size(1), 3,
                                        dtype=self.points.dtype, device=self.points.device)

            _Processing.estimate_normals(
                self.points, self.mask, self._normals,
                EstimateNormalsMethod.CentralDifferences)

        return self._normals

    @normals.setter
    def normals(self, normals):
        self._normals = normals

    def unordered_point_cloud(self, world_space=True, compute_normals=True):
        mask = self.mask.flatten()
        if compute_normals:
            normals = self.normals.view(-1, 3)
            normals = normals[mask]
        else:
            normals = None

        pcl = PointCloud(self.points.view(-1, 3)[mask],
                         self.colors.view(-1, 3)[mask]
                         if self.colors is not None else None,
                         normals)

        if world_space and self.rt_cam is not None:
            pcl = pcl.transform(
                self.rt_cam.cam_to_world.to(self.device).float())

        return pcl

    def downsample(self, scale, downsample_xyz_method=DownsampleXYZMethod.Nearest):
        points = downsample_xyz(self.points, self.mask, scale,
                                method=downsample_xyz_method)
        mask = downsample_mask(self.mask, scale)

        normals = None
        if self._normals is not None:
            normals = downsample_xyz(self._normals, self.mask, scale,
                                     normalize=True,
                                     method=downsample_xyz_method)

        colors = self.colors.transpose(
            1, 2).transpose(1, 0).unsqueeze(0).float()
        colors = interpolate(colors, scale_factor=scale, mode='bilinear',
                             align_corners=False)
        colors = colors.squeeze().transpose(0, 1).transpose(2, 1).byte()
        kcam = self.kcam.scaled(scale)

        return FramePointCloud(None, mask, kcam, rt_cam=self.rt_cam,
                               points=points, normals=normals,
                               colors=colors)

    def pyramid(self, scales):
        pyramid = []
        curr = self
        for scale in scales:
            if scale < 1.0:
                curr = curr.downsample(scale)
            pyramid.append(curr)

        pyramid.reverse()
        return pyramid

    def to(self, dst):
        if isinstance(dst, torch.dtype):
            return FramePointCloud(
                (self.image_points.to(dst) if self.image_points is not None else None),
                self.mask,
                self.kcam.to(dst),
                self.rt_cam.to(dst),
                self._points.to(dst) if self._points is not None else None,
                self._normals.to(dst) if self._normals is not None else None,
                self.colors if self.colors is not None else None)
        else:
            return FramePointCloud(
                (self.image_points.to(dst)
                 if self.image_points is not None else None),
                self.mask.to(dst),
                self.kcam.to(dst),
                self.rt_cam,
                self._points.to(dst) if self._points is not None else None,
                self._normals.to(dst) if self._normals is not None else None,
                self.colors.to(dst) if self.colors is not None else None)

    def __getitem__(self, *slices):
        slices = slices[0]
        return FramePointCloud(self.image_points[slices],
                               self.mask[slices], self.kcam, rt_cam=self.rt_cam,
                               points=self.points[slices],
                               normals=self.normals[slices],
                               colors=self.colors[slices])

    @property
    def width(self):
        return self.mask.size(1)

    @property
    def height(self):
        return self.mask.size(0)

    @property
    def device(self):
        return self.mask.device

    def plot_debug(self, show=True):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 3, 1)
        plt.imshow(self.points[:, :, 0].cpu().numpy())

        plt.subplot(2, 3, 2)
        plt.imshow(self.points[:, :, 1].cpu().numpy())

        plt.subplot(2, 3, 3)
        plt.imshow(self.points[:, :, 2].cpu().numpy())

        plt.subplot(2, 3, 4)
        plt.imshow(self.mask.cpu().numpy())

        plt.subplot(2, 3, 5)
        plt.imshow(self.colors.cpu().numpy())

        plt.subplot(2, 3, 6)
        plt.imshow(self.normals.cpu().numpy())

        if show:
            plt.show()
