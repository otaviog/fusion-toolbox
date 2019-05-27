"""Dataset viewer
"""

from collections import deque

import cv2
import numpy as np
import torch
from matplotlib.pyplot import get_cmap

import tenviz

from fiontb.frame import FramePointCloud
from fiontb.camera import Homogeneous

_CAM_HAND_MATRIX = np.eye(4)
_CAM_HAND_MATRIX[2, 2] = -1


class DatasetViewer:
    """Viewer of RGB-D datasets. It shows the image, depth, camera point
    cloud and accumulated world points.

    """

    def __init__(self, dataset, title="Dataset"):
        self.dataset = dataset
        self.title = title
        self.show_mask = False
        self.last_proc_data = {'idx': -1}

        self.context = tenviz.Context(640, 480)

        with self.context.current():
            axis = tenviz.create_axis_grid(-1, 1, 10)

        self.cam_viewer = self.context.viewer(
            [axis], tenviz.CameraManipulator.TrackBall)
        self.cam_viewer.set_title("{}: camera space".format(title))
        self.tv_camera_pcl = None

        self.wcontext = tenviz.Context(640, 480)
        with self.wcontext.current():
            pass

        self.world_viewer = self.wcontext.viewer(
            [], tenviz.CameraManipulator.WASD)
        self.world_viewer.set_title("{}: world space".format(title))
        self.pcl_deque = deque()

        self.visited_idxs = set()

    def _update_world(self, idx, rt_cam, cam_space, colors, cam_proj):
        if idx in self.visited_idxs:
            return

        self.visited_idxs.add(idx)

        world_space = Homogeneous(rt_cam.cam_to_world) @ cam_space

        with self.wcontext.current():
            pcl = tenviz.create_point_cloud(torch.from_numpy(world_space).float(),
                                            torch.from_numpy(colors).byte())
            self.world_viewer.get_scene().add(pcl)
            vcam = tenviz.create_virtual_camera(
                cam_proj, rt_cam.cam_to_world @ _CAM_HAND_MATRIX)
            self.world_viewer.get_scene().add(vcam)

            self.pcl_deque.append((pcl, vcam))

        if not self.visited_idxs:
            self.world_viewer.reset_view()

        if len(self.pcl_deque) > 50:
            with self.wcontext.current():
                oldest_pcl, oldest_cam = self.pcl_deque.popleft()
                self.world_viewer.get_scene().erase(oldest_pcl)
                self.world_viewer.get_scene().erase(oldest_cam)
                oldest_pcl = oldest_cam = None

        self.wcontext.collect_garbage()

    def _set_model(self, frame, idx):
        finfo = frame.info
        cmap = get_cmap('viridis', finfo.depth_max)

        depth_img = (frame.depth_image / finfo.depth_max)
        depth_img = cmap(depth_img)
        depth_img = depth_img[:, :, 0:3]
        depth_img = (depth_img*255).astype(np.uint8)

        rgb_img = cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2BGR)

        self.last_proc_data = {
            'idx': idx,
            'depth_img': depth_img,
            'rgb_img': rgb_img,
            'fg_mask': frame.fg_mask
        }

        with self.context.current():
            self.cam_viewer.get_scene().erase(self.tv_camera_pcl)

        pcl = FramePointCloud(frame).unordered_point_cloud(world_space=False)
        cam_space = pcl.points

        _cam_matrix_inv_y = _CAM_HAND_MATRIX.copy()
        _cam_matrix_inv_y[1, 1] *= -1

        with self.context.current():
            self.tv_camera_pcl = tenviz.create_point_cloud(
                torch.from_numpy(Homogeneous(_cam_matrix_inv_y)
                                 @ cam_space).float(),
                torch.from_numpy(pcl.colors))
        self.cam_viewer.get_scene().add(self.tv_camera_pcl)

        cam_proj = tenviz.projection_from_kcam(
            finfo.kcam.matrix, 0.5, cam_space[:, 2].max())

        self.cam_viewer.reset_view()

        if finfo.rt_cam is not None:
            self._update_world(idx, finfo.rt_cam, cam_space,
                               pcl.colors, cam_proj)

        self.context.collect_garbage()

    def _update_canvas(self, _):
        idx = cv2.getTrackbarPos("pos", self.title)

        if self.last_proc_data['idx'] != idx:
            frame = self.dataset[idx]
            self._set_model(frame, idx)

        proc_data = self.last_proc_data
        alpha = cv2.getTrackbarPos("oppacity", self.title) / 100.0
        canvas = cv2.addWeighted(proc_data['depth_img'], alpha,
                                 proc_data['rgb_img'], 1.0 - alpha, 0.0)

        fg_mask = proc_data['fg_mask']
        if self.show_mask and fg_mask is not None:
            canvas[~fg_mask] = (0, 0, 0)

        cv2.imshow(self.title, canvas)

    def run(self):
        """Show the viewer and block until user exit. Keys:
        * 'q': quit
        * 'm': toggle masking (when the dataset has it)
        """

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("pos", self.title, 0, len(
            self.dataset) - 1, self._update_canvas)
        cv2.createTrackbar("oppacity", self.title, 50, 100,
                           self._update_canvas)

        while True:
            self._update_canvas(None)
            key = cv2.waitKey(1)

            self.cam_viewer.draw(0)
            self.world_viewer.draw(0)

            if key == 27:
                break

            key = chr(key & 0xff).lower()

            if key == 'q':
                break
            elif key == 'm':
                self.show_mask = not self.show_mask

        cv2.destroyWindow(self.title)
