"""Shared datatypes between dataset and volume fusion.
"""
import numpy as np


class Snapshot:
    def __init__(self, depth_image, kcam=None, depth_scale=1.0, depth_bias=0.0,
                 rgb_image=None, rt_cam=None, rgb_kcam=None, fg_mask=None):
        self.depth_image = depth_image
        
        self.depth_scale = depth_scale
        self.depth_bias = 0.0

        xs, ys = np.meshgrid(np.arange(depth_image.shape[1]),
                             np.arange(depth_image.shape[0]))
        points = np.dstack([xs, ys, depth_image])

        self.img_points = points.reshape(
            (points.shape[0]*points.shape[1], 3, 1))

        self.fg_mask = fg_mask
        if fg_mask is not None:
            fg_mask = fg_mask.flatten()
            self.img_points = self.img_points[fg_mask]

        if kcam is not None:
            self.cam_points = kcam.backproject(self.img_points)
        else:
            self.cam_points = None

        if rt_cam is not None:
            self.world_points = rt_cam.transform_cam_to_world(
                self.cam_points)
        else:
            self.world_points = None

        self.rgb_image = rgb_image
        if rgb_image is not None:
            self.colors = rgb_image.reshape(
                (rgb_image.shape[0]*rgb_image.shape[1], 3))

            if fg_mask is not None:
                self.colors = self.colors[fg_mask]

        self.kcam = kcam
        self.rt_cam = rt_cam

        self.rgb_kcam = rgb_kcam
