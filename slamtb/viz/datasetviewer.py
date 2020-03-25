"""Dataset viewer
"""

from collections import deque

import cv2
import numpy as np
import torch
from matplotlib.pyplot import get_cmap

import tenviz

from slamtb.frame import FramePointCloud
from slamtb.camera import RigidTransform


class DatasetViewer:
    """A Viewer for RGB-D datasets. It'll show the RGB, depth, camera's
    point cloud and accumulated world space point clouds.

    """

    def __init__(self, dataset, title="Dataset", max_pcls=50,
                 width=640, height=480, invert=False,
                 camera_view=True, trajectory_cmap="Blues", show_grid=True):
        """Setups the dataset. Call `run` to show.

        Args:

            dataset (List[:obj:``): Any dataset, that is, classes that
        function like a list of frames.

        """

        self.dataset = dataset
        self.title = title
        self.show_segmentation = False
        self.last_proc_data = {'idx': -1}
        self.invert = invert

        self.cam_context = None
        if camera_view:
            self.cam_context = tenviz.Context(width, height)

            with self.cam_context.current():
                # TODO: fix tensorviz creating viewer without `current()`
                pass
            self.cam_viewer = self.cam_context.viewer(
                [], tenviz.CameraManipulator.WASD)
            self.cam_viewer.title = "{}: camera's point cloud".format(title)
            self.tv_camera_pcl = None

        self.wcontext = tenviz.Context(width, height)
        scene = []
        with self.wcontext.current():
            if show_grid:
                scene = [tenviz.nodes.create_axis_grid(-1, 1, 10)]

        self.world_viewer = self.wcontext.viewer(
            scene, tenviz.CameraManipulator.WASD)
        self.world_viewer.title = "{}: accumulated point clouds".format(title)
        self.pcl_deque = deque()

        self.visited_idxs = set()

        self._show_cams = True
        self.max_pcls = max_pcls

        self._trajectory_cmap = get_cmap(trajectory_cmap, len(dataset))

    def _set_model(self, frame, idx):
        finfo = frame.info
        cmap = get_cmap('viridis', int(finfo.depth_max))

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
            'seg_img': frame.seg_image
        }

        pcl = FramePointCloud.from_frame(
            frame).unordered_point_cloud(world_space=False, compute_normals=False)
        cam_space = pcl.points

        if self.cam_context is not None:
            with self.cam_context.current():
                self.cam_viewer.get_scene().erase(self.tv_camera_pcl)

            with self.cam_context.current():
                self.tv_camera_pcl = tenviz.nodes.PointCloud(
                    cam_space, pcl.colors)
            self.cam_viewer.get_scene().add(self.tv_camera_pcl)
            # self.cam_viewer.reset_view()
            self.cam_context.collect_garbage()

        cam_proj = tenviz.Projection.from_intrinsics(
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
            pcl = tenviz.nodes.PointCloud(world_space, colors)
            self.world_viewer.get_scene().add(pcl)

            vcam_color = self._trajectory_cmap(idx)[:3]
            vcam = tenviz.nodes.create_virtual_camera(
                cam_proj,
                np.linalg.inv(rt_cam.opengl_view_cam),
                color=vcam_color, line_width=1, show_axis=False)
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

        seg_image = proc_data['seg_img']
        if self.show_segmentation and seg_image is not None:
            cmap = get_cmap("tab20", 1000)
            rgb_img = (cmap(seg_image)[:, :, :3]*255).astype(np.uint8)
        else:
            rgb_img = proc_data['rgb_img']

        alpha = cv2.getTrackbarPos("oppacity", self.title) / 100.0
        canvas = cv2.addWeighted(proc_data['depth_img'], alpha,
                                 rgb_img, 1.0 - alpha, 0.0)

        cv2.imshow(self.title, canvas)

    def _mouse_click(self, event, x, y, *args):
        if event != cv2.EVENT_LBUTTONUP:
            return
        seg_image = self.last_proc_data['seg_img']
        if seg_image is not None:
            print("Segmentation value at x {} y {}: {}".format(
                x, y, seg_image[y, x]))

    def run(self):
        """Show the viewer and block until user exit. Keys:

        * 'q': quit;
        * 'm': toggle masking (when the dataset has it);
        * 'c': toggle camera frustums.
        """

        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("pos", self.title, 0, len(
            self.dataset) - 1, self._update_canvas)
        cv2.createTrackbar("oppacity", self.title, 50, 100,
                           self._update_canvas)
        cv2.setMouseCallback(self.title, self._mouse_click)
        self._update_canvas(None)
        while True:
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
                    self.show_segmentation = not self.show_segmentation
                    self._update_canvas(None)
                elif key == 'c':
                    self._show_cams = not self._show_cams
                    for _, vcam in self.pcl_deque:
                        vcam.visible = self._show_cams
            if quit_loop:
                break

        cv2.destroyWindow(self.title)
