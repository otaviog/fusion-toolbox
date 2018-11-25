"""Shared datatypes between dataset and volume fusion.
"""
import numpy as np


class Snapshot:
    def __init__(self, depth_image, color_image=None, intr_cam=None, extr_cam=None):
        self.depth_image = depth_image

        xs, ys = np.meshgrid(np.arange(depth_image.shape[1]),
                             np.arange(depth_image.shape[0]))
        points = np.dstack([xs, ys, depth_image])

        self.img_points = points.reshape(
            (points.shape[0]*points.shape[1], 3, 1))

        if intr_cam is not None:
            self.cam_points = intr_cam.unproject_image_to_cam(self.img_points)
        else:
            self.cam_points = None

        if extr_cam is not None:
            self.world_points = extr_cam.transform_cam_to_world(
                self.cam_points)
        else:
            self.world_points = None

        self.color_image = color_image
        if color_image is not None:
            self.colors = color_image.reshape(
                (color_image.shape[0]*color_image.shape[1], 3))

        self.kcam = intr_cam
        self.rtcam = extr_cam
