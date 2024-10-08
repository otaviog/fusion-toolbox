"""Frame related data structures.
"""
import textwrap

import torch
from torch.nn.functional import interpolate
import cv2
import numpy as np

from slamtb.processing import (
    downsample_xyz, downsample_mask, DownsampleXYZMethod, erode_mask)
from slamtb._cslamtb import (Processing as _Processing, EstimateNormalsMethod)
from slamtb._utils import ensure_torch, depth_image_to_uvz

from .camera import KCamera, RTCamera
from .pointcloud import PointCloud


class FrameInfo:
    """A common description for frames. Holds information about camera
    parameters, depth scaling, and time.

    Attributes:

        kcam (:obj:`slamtb.camera.KCamera`): The intrinsic camera.

        depth_scale (float): Scaling for raw depth values directly
         from files.

        depth_bias (float): Constant add to the raw depth values.

        depth_max (float): Sensor maximum depth value.

        rt_cam (:obj:`slamtb.camera.RTCamera`): The extrinsic camera.

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
        """Creates a from FTB's JSON representation.

        Args:

            json (dict): JSON dict.
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
        """Converts the frame info to its FTB JSON dictionary representation.

        Returns:
            (dict): JSON dictionary.

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
        """Creates a copy of the instance.

        Returns:
            (:obj:`FrameInfo`): Copy.
        """

        return FrameInfo(
            self.kcam.clone(), self.depth_scale, self.depth_bias,
            self.depth_max, self.timestamp,
            None if self.rgb_kcam is None else self.rgb_kcam.clone(),
            None if self.rt_cam is None else self.rt_cam.clone())

    def __str__(self):
        return ("Depth intrinsics: {{{self.kcam}}}"
                ", depth scale: {self.depth_scale}"
                ", depth bias: {self.depth_bias}"
                ", timestamp: {self.timestamp}"
                ", Rigid Transformation: {{{self.rt_cam}}}").format(self=self)

    def __repr__(self):
        return str(vars(self))


class Frame:
    """A RGB-D frame, either outputted from a sensor or dataset.

    Attributes:

        info (:obj:`FrameInfo`): Information about depth and camera parameters.

        depth_image (:obj:`numpy.ndarray`): Depth image of size (H x
         W) and int32 type.

        rgb_image (:obj:`numpy.ndarray`, optional): RGB image of size (H x W x
         3) and uint8 type.

        seg_image (:obj:`numpy.ndarray`, optional): Segmentation image
         of size (H x W) and int16 type. Value 0 means background.

        normal_image (:obj:`numpy.ndarray`, optional): Image containing
         per point normal vectors. Size is (H x W x 3) with float type.

    """

    def __init__(self, info: FrameInfo, depth_image, rgb_image=None,
                 seg_image=None, normal_image=None):
        self.info = info
        self.info.kcam.image_size = (depth_image.shape[1],
                                     depth_image.shape[0])
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.seg_image = seg_image
        self.normal_image = normal_image

    def clone(self, shallow=False):
        """Copy the instance.

        Args:

            shallow (bool, optional): If `False`, then the images are
             also cloned. Otherwise, it'll just create another
             instance of the frame.

        Returns:
            (:obj:`Frame`): Frame copy.
        """

        def _clone(tensor):
            if torch.is_tensor(tensor):
                return tensor.cpu().clone().numpy()
            return tensor.copy()
        
        if not shallow:
            return Frame(self.info.clone(), self.depth_image.copy(),
                         None if self.rgb_image is None else self.rgb_image.copy(),
                         None if self.seg_image is None else self.seg_image.copy(),
                         None if self.normal_image is None else _clone(self.normal_image))

        return Frame(self.info, self.depth_image,
                     self.rgb_image, self.seg_image, self.normal_image)

    def scaled(self, xscale, yscale=None, interpolation=cv2.INTER_LINEAR):
        if yscale is None:
            yscale = xscale

        rgb_image = None
        if self.rgb_image is not None:
            rgb_image = cv2.resize(self.rgb_image, (0, 0),
                                   None, xscale, yscale, interpolation)

        depth_image = cv2.resize(
            self.depth_image.astype(np.uint16), (0, 0), None, xscale, yscale,
            interpolation).astype(np.int32)

        seg_image = None
        if self.seg_image is not None:
            seg_image = cv2.resize(self.seg_image, (0, 0),
                                   None, xscale, yscale, cv2.INTER_NEAREST)

        normal_image = None
        if self.normal_image is not None:
            normal_image = cv2.resize(
                self.normal_image, (0, 0), None, xscale, yscale, cv2.INTER_LINEAR)

        info = FrameInfo(
            kcam=self.info.kcam.scaled(xscale, yscale),
            depth_scale=self.info.depth_scale,
            depth_bias=self.info.depth_bias,
            depth_max=self.info.depth_max,
            timestamp=self.info.timestamp,
            rgb_kcam=(self.info.rgb_kcam.scaled(xscale, yscale)
                      if self.info.rgb_kcam is not None else None),
            rt_cam=self.info.rt_cam)

        return Frame(info, depth_image, rgb_image, seg_image, normal_image)

    def __str__(self):
        has_depth = self.depth_image is not None
        has_rgb = self.rgb_image is not None
        has_seg = self.seg_image is not None
        has_normal = self.normal_image is not None

        return (
            "Frame with shape ({self.depth_image.shape[0]}x{self.depth_image.shape[1]})"
            " depth: {has_depth}, RGB: {has_rgb}, segmentation: {has_seg}, Normal: {has_normal}"
        ).format(self=self, has_depth=has_depth, has_rgb=has_rgb,
                 has_seg=has_seg, has_normal=has_normal)

        s = """\
        Frame with shape: {self.depth_image.shape[0]}, {self.depth_image.shape[1]}\
        with depth: {has_depth}, RGB: {has_rgb}, segmentation: {has_seg}, Normal: {has_normal}
              info: {self.info}
        """.format(self=self, has_depth=has_depth, has_rgb=has_rgb,
                   has_seg=has_seg, has_normal=has_normal)

        return textwrap.dedent(s)


class FramePointCloud:
    """A point cloud still embedded on its frame. This representation is
    useful for retrieving point cloud data by pixel coordinates.

    Attributes:

        image_points (:obj:`torch.Tensor`): Image of U, V and depth values (H
         x W x 3). Float 32 type.

        mask (:obj:`torch.Tensor`): Valid mask (bool) of uv coordinates (H x W).

        kcam (:obj:`slamtb.camera.KCamera`): Camera intrinsic parameters.

        rt_cam (:obj:`slamtb.camera.RTCamera`): Camera extrinsic parameters.

        points (:obj:`torch.Tensor`): Image of 3D points (H x W x 3) located
         on the camera space. Float32 type.

        colors (:obj:`torch.Tensor`): Image of point colors (H x W x
         3). Uint8 type.

        normals (:obj:`torch.Tensor`): Image of normal vectors (H x W x
         3). Float32 type.

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
    def from_frame(cls, frame: Frame, ignore_seg_background=False):
        """Create the frame point cloud from a frame.

        Args:

            frame (:obj:`Frame`): Frame.

            ignore_seg_background (bool, optional). If `True` and
             frame has segmentation, then it'll discard region that are
             background (seg_image == 0). Default is `False`.

        """

        depth_image = ensure_torch(frame.depth_image)

        image_points = depth_image_to_uvz(depth_image, frame.info)

        mask = (depth_image > 0)
        mask = erode_mask(mask)

        if frame.seg_image is not None and ignore_seg_background:
            mask = torch.logical_and(
                torch.from_numpy(frame.seg_image > 0), mask)

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
        """Image of 3D points on the camera space.


        Returns: (:obj:`torch.Tensor`): (H x W x 3) 3D points.
        """

        if self._points is None:
            if self.kcam is None:
                raise RuntimeError("Frame doesn't have intrinsics camera")

            self._points = self.kcam.backproject(
                self.image_points.reshape(-1, 3)).reshape(self.image_points.shape)
        return self._points

    @property
    def normals(self):
        """Image of 3D normal vectors. It'll compute normals by the central
        differences method if not computed before.

        Returns:
            (:obj:`torch.Tensor`): (H x W x 3) per point normal vectors.

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
        """Overwrites the normals vectors.

        Args:

            normals (:obj:`torch.Tensor`): A normal vector image (H x
             W x 3).

        """

        self._normals = normals

    def unordered_point_cloud(self, world_space=True, compute_normals=True):
        """Converts into a sparse point cloud.

        Args:

            world_space (bool, optional): If `True`, then the 3d point
             are transformed into world space. Default is `True`.

            compute_normals (bool, optional): If `True` the normals
             are included on the returned point cloud.

        """

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

    def downsample(self, scale, downsample_xyz_method=DownsampleXYZMethod.Nearest, colored=True):
        """Downsample the point cloud.

        Args:

            scale (float): Scaling.

            downsample_xyz_method (DownsampleXYZMethod): Which
             sampling method for interpolating points and normals.

        Returns:
            (:obj:`FramePointCloud`): Down-sampled point cloud.
        """

        points = downsample_xyz(self.points, self.mask, scale,
                                method=downsample_xyz_method)
        normals = None
        if self._normals is not None:
            normals = downsample_xyz(self._normals, self.mask, scale,
                                     normalize=True,
                                     method=downsample_xyz_method)

        mask = downsample_mask(self.mask, scale)

        colors = None
        if colored:
            colors = self.colors.permute(2, 0, 1).unsqueeze(0).float()
            colors = interpolate(colors, scale_factor=scale, mode='bilinear',
                                 align_corners=False)
            colors = colors.squeeze(0).permute(1, 2, 0).byte()
        kcam = self.kcam.scaled(scale)

        return FramePointCloud(None, mask, kcam, rt_cam=self.rt_cam,
                               points=points, normals=normals,
                               colors=colors)

    def pyramid(self, scales, downsample_xyz_method=DownsampleXYZMethod.Nearest,
                colored=False):
        """Create a multiple scale pyramid for this point cloud.

        Args:

            scales (List[float]): Decreasing scale factors. No scaling
             is applied for values greater than one, repeating the
             current point cloud.

            downsample_xyz_method (DownsampleXYZMethod): Which
             sampling method for interpolating points and normals.

        Returns:
            (List[:obj:`FramePointCloud`]): Downsampled pyramid of point
             clouds. In increasing order of scales.

        """

        pyramid = []
        curr = self
        for scale in scales:
            if scale < 1.0:
                curr = curr.downsample(
                    scale, downsample_xyz_method=downsample_xyz_method,
                    colored=colored)
            pyramid.append(curr)

        pyramid.reverse()
        return pyramid

    # pylint: disable=invalid-name
    def to(self, dst):
        """Change the point cloud device or dtype. Dtype are applied only to
        points and normals.

        Args:

            dst (torch.dtype, torch.device, str): Dtype or torch device.

        Returns:
            (:obj:`FramePointCloud`): converted point cloud.

        """
        if isinstance(dst, torch.dtype):
            return FramePointCloud(
                (self.image_points.to(dst) if self.image_points is not None else None),
                self.mask,
                self.kcam,
                self.rt_cam,
                self._points.to(dst) if self._points is not None else None,
                self._normals.to(dst) if self._normals is not None else None,
                self.colors if self.colors is not None else None)

        return FramePointCloud(
            (self.image_points.to(dst)
             if self.image_points is not None else None),
            self.mask.to(dst),
            self.kcam.clone(),
            self.rt_cam.clone(),
            self._points.to(dst) if self._points is not None else None,
            self._normals.to(dst) if self._normals is not None else None,
            self.colors.to(dst) if self.colors is not None else None)

    def __getitem__(self, *slices):
        slices = slices[0]
        return FramePointCloud(self.image_points[slices],
                               self.mask[slices], self.kcam, rt_cam=self.rt_cam,
                               points=self.points[slices],
                               normals=self.normals[slices] if self._normals is not None else None,
                               colors=self.colors[slices])

    @property
    def width(self):
        """Frame width (int).
        """
        return self.mask.size(1)

    @property
    def height(self):
        """Frame height (int).
        """
        return self.mask.size(0)

    @property
    def device(self):
        """Torch device (str).
        """
        return self.mask.device

    def plot_debug(self, show=True):
        """Debug ploting.
        """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        plt.figure()
        plt.subplot(2, 3, 1)
        plt.title("X-values")
        plt.imshow(self.points[:, :, 0].cpu().numpy())

        plt.subplot(2, 3, 2)
        plt.title("Y-values")
        plt.imshow(self.points[:, :, 1].cpu().numpy())

        plt.subplot(2, 3, 3)
        plt.title("Z-values")
        plt.imshow(self.points[:, :, 2].cpu().numpy())

        plt.subplot(2, 3, 4)
        plt.title("Mask")
        plt.imshow(self.mask.cpu().numpy())

        plt.subplot(2, 3, 5)
        plt.title("Colors")
        plt.imshow(self.colors.cpu().numpy())

        plt.subplot(2, 3, 6)
        plt.title("Normals")
        plt.imshow(self.normals.cpu().numpy())

        if show:
            plt.show()
