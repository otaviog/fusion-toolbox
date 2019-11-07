"""Dataset viewer
"""

from collections import deque

import cv2
import numpy as np
import torch
from matplotlib.pyplot import get_cmap

import tenviz

from fiontb.frame import FramePointCloud
from fiontb.camera import RigidTransform


class DatasetViewer:
    """Viewer of RGB-D datasets. It shows the image, depth, camera point
    cloud and accumulated world points.
    """

    def __init__(self, dataset, title="Dataset", max_pcls=50, invert=False,
                 camera_view=True):
        self.dataset = dataset
        self.title = title
        self.show_mask = False
        self.last_proc_data = {'idx': -1}
        self.invert = invert

        self.cam_context = None
        if camera_view:
            self.cam_context = tenviz.Context(640, 480)

            with self.cam_context.current():
                # TODO: fix tensorviz creating viewer without `current()`
                pass
            self.cam_viewer = self.cam_context.viewer(
                [], tenviz.CameraManipulator.TrackBall)
            self.cam_viewer.title = "{}: camera space".format(title)
            self.tv_camera_pcl = None

        self.wcontext = tenviz.Context(640, 480)
        with self.wcontext.current():
            axis = tenviz.create_axis_grid(-1, 1, 10)

        self.world_viewer = self.wcontext.viewer(
            [axis], tenviz.CameraManipulator.WASD)
        self.world_viewer.title = "{}: world space".format(title)
        self.pcl_deque = deque()

        self.visited_idxs = set()

        self._show_cams = True
        self.max_pcls = max_pcls

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

        pcl = FramePointCloud.from_frame(
            frame).unordered_point_cloud(world_space=False, compute_normals=False)
        cam_space = pcl.points

        if self.cam_context is not None:
            with self.cam_context.current():
                self.cam_viewer.get_scene().erase(self.tv_camera_pcl)

            with self.cam_context.current():
                self.tv_camera_pcl = tenviz.create_point_cloud(
                    cam_space, pcl.colors)
            self.cam_viewer.get_scene().add(self.tv_camera_pcl)
            self.cam_viewer.reset_view()
            self.cam_context.collect_garbage()

        cam_proj = tenviz.projection_from_kcam(
            finfo.kcam.matrix, 0.5, cam_space[:, 2].max())

        if finfo.rt_cam is not None:
            self._update_world(idx, finfo.rt_cam,
                               cam_space,
                               pcl.colors, cam_proj)

    def _update_world(self, idx, rt_cam, cam_space, colors, cam_proj):
        if idx in self.visited_idxs:
            return

        self.visited_idxs.add(idx)

        rt_cam.matrix = rt_cam.matrix.float()
        invert_cam = torch.eye(4, dtype=torch.float)

        if self.invert:
            invert_cam[0, 0] *= -1
            invert_cam[1, 1] *= -1

        world_space = RigidTransform(
            invert_cam @ rt_cam.cam_to_world) @ cam_space

        with self.wcontext.current():
            pcl = tenviz.create_point_cloud(world_space, colors)
            self.world_viewer.get_scene().add(pcl)
            vcam = tenviz.create_virtual_camera(
                cam_proj,
                np.linalg.inv(rt_cam.opengl_view_cam))
            vcam.visible = self._show_cams
            self.world_viewer.get_scene().add(vcam)

            self.pcl_deque.append((pcl, vcam))

        if not self.visited_idxs:
            self.world_viewer.reset_view()

        if len(self.pcl_deque) > self.max_pcls:
            with self.wcontext.current():
                oldest_pcl, oldest_cam = self.pcl_deque.popleft()
                self.world_viewer.get_scene().erase(oldest_pcl)
                self.world_viewer.get_scene().erase(oldest_cam)
                oldest_pcl = oldest_cam = None

        self.wcontext.collect_garbage()

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
            cv_key = cv2.waitKey(1)

            if cv_key == 27:
                break

            if cv_key < 0:
                cv_key = 0

            quit_loop = False
            keys = [cv_key, self.world_viewer.wait_key(0)]
            if self.cam_context is not None:
                keys.append(self.cam_viewer.wait_key(0))

            for key in keys:
                if key < 0:
                    quit_loop = True

                key = chr(key & 0xff).lower()

                if key == 'q':
                    quit_loop = True
                elif key == 'm':
                    self.show_mask = not self.show_mask
                elif key == 'c':
                    self._show_cams = not self._show_cams
                    for _, vcam in self.pcl_deque:
                        vcam.visible = self._show_cams
            if quit_loop:
                break
        cv2.destroyWindow(self.title)
