import cv2
import pyquaternion
import numpy as np
import torch

import tenviz
import tenviz.io
import tenviz.geometry


class ReconstructionViewer:
    _GT_BRIGHTNESS = "Ground truth brightness"
    _REC_BRIGHTNESS = "Reconstruction brightness"

    def __init__(self, gt_geo, rec_geo, title="Viewer"):
        self.context = tenviz.Context(640, 480)

        with self.context.current():
            gt_geo = gt_geo
            if isinstance(gt_geo, tenviz.geometry.Geometry):
                if gt_geo.faces is not None:
                    self.gt_geo = tenviz.create_mesh(
                        gt_geo.verts, gt_geo.faces, normals=gt_geo.normals)
                else:
                    self.gt_geo = tenviz.create_point_cloud(
                        gt_geo.verts,
                        gt_geo.colors)
                    self.gt_geo.style.point_size = 2
            else:
                self.gt_geo = tenviz.create_point_cloud(
                    gt_geo.points,
                    gt_geo.colors)
                self.gt_geo.style.point_size = 2

            self.rec_pcl = tenviz.create_point_cloud(rec_geo.verts,
                                                     rec_geo.colors)
            self.rec_pcl.point_size = 2

            grid = tenviz.create_axis_grid(-5, 5, 10)

        self.viewer = self.context.viewer([self.rec_pcl, self.gt_geo, grid],
                                          tenviz.CameraManipulator.WASD)

        self.title = title
        cv2.namedWindow(self.title)
        cv2.createTrackbar(self._GT_BRIGHTNESS, self.title,
                           100, 100, self._update_scene)
        cv2.createTrackbar(self._REC_BRIGHTNESS, self.title,
                           100, 100, self._update_scene)

    def _update_scene(self, _):
        if hasattr(self.gt_geo, "transparency"):
            self.gt_geo.transparency = cv2.getTrackbarPos(
                self._GT_BRIGHTNESS, self.title) / 100.0

        self.rec_pcl.transparency = cv2.getTrackbarPos(
            self._REC_BRIGHTNESS, self.title) / 100.0

    def run(self):
        while True:
            self.viewer.draw(0)

            key = cv2.waitKey(5) & 0xff
            if key == 27:
                break

            key = chr(key)

            if key == '1':
                self.gt_geo.visible = not self.gt_geo.visible

            if key == '2':
                self.rec_pcl.visible = not self.rec_pcl.visible

        cv2.destroyWindow(self.title)
        self.viewer = None
        self.context = None


class AlignTool(ReconstructionViewer):
    _XROT = "X rot"
    _YROT = "Y rot"
    _ZROT = "Z rot"
    _XT = "X"
    _YT = "Y"
    _ZT = "Z"

    _TSIZE = 2000

    def __init__(self, gt_geo, rec_geo, title="Align tool"):
        super().__init__(gt_geo, rec_geo, title)
        self.transformation = torch.eye(4)

        cv2.createTrackbar(AlignTool._XROT, self.title,
                           0, 360, self._update_align)
        cv2.createTrackbar(AlignTool._YROT, self.title,
                           0, 360, self._update_align)
        cv2.createTrackbar(AlignTool._ZROT, self.title,
                           0, 360, self._update_align)
        cv2.createTrackbar(AlignTool._XT, self.title,
                           int(AlignTool._TSIZE/2),
                           AlignTool._TSIZE,
                           self._update_align)
        cv2.createTrackbar(AlignTool._YT, self.title,
                           int(AlignTool._TSIZE/2),
                           AlignTool._TSIZE,
                           self._update_align)
        cv2.createTrackbar(AlignTool._ZT, self.title,
                           int(AlignTool._TSIZE/2),
                           AlignTool._TSIZE,
                           self._update_align)

    def _update_align(self, _):
        rotx = pyquaternion.Quaternion(
            axis=[1, 0, 0],
            angle=np.deg2rad(cv2.getTrackbarPos(self._XROT, self.title)))

        roty = pyquaternion.Quaternion(
            axis=[0, 1, 0],
            angle=np.deg2rad(cv2.getTrackbarPos(self._YROT, self.title)))
        rotz = pyquaternion.Quaternion(
            axis=[0, 0, 1],
            angle=np.deg2rad(cv2.getTrackbarPos(self._ZROT, self.title)))

        rot = (rotx*roty*rotz).transformation_matrix
        trans = np.eye(4)
        trans[0, 3] = (cv2.getTrackbarPos(
            self._XT, self.title) / self._TSIZE)*20 - 10
        trans[1, 3] = (cv2.getTrackbarPos(
            self._YT, self.title) / self._TSIZE)*20 - 10
        trans[2, 3] = (cv2.getTrackbarPos(
            self._ZT, self.title) / self._TSIZE)*20 - 10

        self.transformation = torch.from_numpy(trans @ rot).float()
        self.rec_pcl.set_transform(self.transformation)
