"""Shared datatypes between dataset and volume fusion.
"""
import numpy as np


class Snapshot:
    """RGB-D view snapshot.

    Attributes:

        depth_image (:obj:`numpy.ndarray`): Original dataset depth image.

        kcam (:obj:`fiontb.Camera`): Camera intrinsic.
    """

    def __init__(self, depth_image, kcam=None,
                 depth_scale=1.0, depth_bias=0.0, depth_max=4500,  # From Kinect
                 rgb_image=None, rt_cam=None, rgb_kcam=None, fg_mask=None,
                 timestamp=None):
        self.depth_image = depth_image

        self.depth_scale = depth_scale
        self.depth_bias = depth_bias
        self.depth_max = depth_max

        self.kcam = kcam
        self.rgb_kcam = rgb_kcam
        self.rt_cam = rt_cam
        self.timestamp = timestamp

        xs, ys = np.meshgrid(np.arange(depth_image.shape[1]),
                             np.arange(depth_image.shape[0]))
        points = np.dstack(
            [xs, ys, self.depth_scale*depth_image + self.depth_bias])

        self.img_points = points.reshape(
            (points.shape[0]*points.shape[1], 3, 1))

        depth_mask = self.depth_image > 0
        if fg_mask is not None:
            self.fg_mask = np.logical_and(fg_mask, depth_mask)
        else:
            self.fg_mask = depth_mask

        self.img_points = self.img_points[self.fg_mask.flatten()]

        self.rgb_image = rgb_image
        if rgb_image is not None:
            self.colors = rgb_image.reshape(
                (rgb_image.shape[0]*rgb_image.shape[1], 3))

            self.colors = self.colors[self.fg_mask.flatten()]

    def get_cam_points(self):
        if self.kcam is not None:
            return self.kcam.backproject(self.img_points)
    
    def get_world_points(self):
        if self.rt_cam is not None:
            return self.rt_cam.transform_cam_to_world(self.get_cam_points())
