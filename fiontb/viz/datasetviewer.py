"""Dataset viewer
"""

from collections import deque

import cv2
import numpy as np
from matplotlib.pyplot import get_cmap

import tenviz

from fiontb.frame import FramePoints
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

        self.cam_viewer = self.context.viewer()
        self.cam_viewer.set_title("{}: camera space".format(title))

        self.viewer_world = _Viewer(640, 480)
        self.viewer_world.view.set_title("{}: world space".format(title))
        self.viewer_world.ctx.add_axis_grid(-10, 10, 1)
        self.visited_idxs = set()

        self.pcl_deque = deque()
        self.visited_idxs = set()

    def _update_world(self, idx, rt_cam, cam_space, colors, cam_proj):
        if idx in self.visited_idxs:
            return

        world_space = Homogeneous(rt_cam.cam_to_world) @ cam_space

        pcl = self.viewer_world.ctx.add_point_cloud(
            world_space, colors)

        cam = self.viewer_world.ctx.add_camera(
            cam_proj, np.matmul(rt_cam.cam_to_world, _CAM_HAND_MATRIX))

        self.pcl_deque.append((pcl, cam))
        self.visited_idxs.add(idx)

        self.viewer_world.view.reset_view()
        if len(self.pcl_deque) > 50:
            oldest_pcl, oldest_cam = self.pcl_deque.popleft()
            self.viewer_world.ctx.erase(oldest_pcl)
            self.viewer_world.ctx.erase(oldest_cam)
            oldest_pcl = oldest_cam = None

    def _update_canvas(self, _):
        idx = cv2.getTrackbarPos("pos", self.title)

        if self.last_proc_data['idx'] != idx:
            frame = self.dataset[idx]
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
            self.cam_viewer.scene.remove(self.tv_camera_pcl)
            self.context.remove(self.tv_camera_pcl)
            
            pcl = FramePoints(frame)

            cam_space = pcl.camera_points

            _cam_matrix_inv_y = _CAM_HAND_MATRIX.copy()
            _cam_matrix_inv_y[1, 1] *= -1

            with self.context.current():
                self.tv_camera_pcl = tenviz.create_point_cloud(
                    Homogeneous(_cam_matrix_inv_y) @ cam_space,
                    pcl.colors)

            cam_proj = tenviz.projection_from_kcam(
                finfo.kcam.matrix, 0.5, cam_space[:, 2].max())

            self.cam_viewer.reset_view()

            if finfo.rt_cam is not None:
                self._update_world(idx, finfo.rt_cam,
                                   cam_space, pcl.colors, cam_proj)

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
