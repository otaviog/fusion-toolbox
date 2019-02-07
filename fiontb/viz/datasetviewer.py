"""Dataset viewer
"""

import cv2
import numpy as np
from matplotlib.pyplot import get_cmap
from pyquaternion import Quaternion
from tqdm import tqdm

import shapelab
import shapelab.io


def _transform(mtx, points):
    points = np.insert(points, 3, 1, axis=1)
    points = np.matmul(mtx, points)
    points = np.delete(points, 3, 1)
    return points


def _gram_schmidt_columns(X):
    # pylint: disable=invalid-name

    Q, _ = np.linalg.qr(X)
    return Q


class DatasetViewer:
    def __init__(self, dataset, title="Dataset"):
        self.dataset = dataset
        self.title = title
        self.show_mask = False
        self.last_proc_data = {'idx': -1}

        self.viewer_cam = shapelab.AsyncViewer.create_default(
            shapelab.RenderConfig(640, 480))
        self.viewer_cam.set_title("{}: camera space".format(title))

        self.viewer_world = shapelab.AsyncViewer.create_default(
            shapelab.RenderConfig(640, 480))
        self.viewer_world.set_title("{}: world space".format(title))
        self.visited_idxs = set()
        self.all_points = []

    def _update_canvas(self, _):
        idx = cv2.getTrackbarPos("pos", self.title)

        if self.last_proc_data['idx'] != idx:
            snap = self.dataset[idx]
            cmap = get_cmap('viridis', snap.depth_max)

            depth_img = (snap.depth_image / snap.depth_max)
            depth_img = cmap(depth_img)
            depth_img = depth_img[:, :, 0:3]
            depth_img = (depth_img*255).astype(np.uint8)

            rgb_img = cv2.cvtColor(
                snap.rgb_image, cv2.COLOR_RGB2BGR)

            self.last_proc_data = {
                'idx': idx,
                'depth_img': depth_img,
                'rgb_img': rgb_img,
                'fg_mask': snap.fg_mask
            }

            self.viewer_cam.clear_scene()

            cam_space = snap.get_cam_points()
            self.viewer_cam.add_point_cloud(cam_space, snap.colors/255)
            self.viewer_cam.reset_view()

            cam_proj = shapelab.projection_from_kcam(snap.kcam.matrix, 0.5, 25)
            self.viewer_cam.set_projection(cam_proj)

            self.viewer_cam.set_view(0.0, 0.0)
            self.viewer_cam.set_camera_matrix(np.eye(4))
            if snap.rt_cam is not None and idx not in self.visited_idxs:
                self.visited_idxs.add(idx)
                rt_cam = snap.rt_cam.matrix

                # rt_cam[0:2, 0:2] = _gram_schmidt_columns(rt_cam[0:2, 0:2])
                conv_handness = np.eye(4)
                # conv_handness[0, 0] *= -1
                conv_handness[2, 2] *= -1

                # world_space = _transform(rt_cam, cam_space)
                # world_space = _transform(conv_handness, cam_space)
                # cam_rt_cam = np.eye(4)

                # cam_rt_cam = np.matmul(rt_cam*conv_handness, cam_rt_cam)
                # rt_cam = rt_cam*conv_handness
                # self.viewer_world.add_camera(
                # cam_proj, rt_cam*conv_handness)
                world_space = snap.get_world_points()
                self.all_points.append(world_space)
                self.viewer_world.add_point_cloud(world_space, snap.colors/255)

                if self.last_proc_data is None:  # First view
                    self.viewer_world.reset_view()

        proc_data = self.last_proc_data
        alpha = cv2.getTrackbarPos("oppacity", self.title) / 100.0
        canvas = cv2.addWeighted(proc_data['depth_img'], alpha,
                                 proc_data['rgb_img'], (1.0 - alpha), 0.0)

        fg_mask = proc_data['fg_mask']
        if self.show_mask and fg_mask is not None:
            canvas[~fg_mask] = (0, 0, 0)

        cv2.imshow(self.title, canvas)

    def run(self):
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("pos", self.title, 0, len(
            self.dataset) - 1, self._update_canvas)
        cv2.createTrackbar("oppacity", self.title, 50, 100,
                           self._update_canvas)

        while True:
            self._update_canvas(None)
            key = cv2.waitKey(-1)
            if key == 27:
                break

            key = chr(key & 0xff).lower()

            if key == 'q':
                break
            elif key == 'm':
                self.show_mask = not self.show_mask
        self.viewer_cam.stop()
        self.viewer_world.stop()
        all_points = np.vstack(self.all_points)
        shapelab.io.write_3dobject("points.off", all_points)


class WorldPointsViewer:
    def __init__(self, dataset, idx_skip=1, max_frames=None):
        self.dataset = dataset
        self.idx_skip = idx_skip
        if max_frames is not None:
            self._max_frames = min(max_frames, len(dataset))
        else:
            self._max_frames = len(dataset)

    def run(self):
        context = shapelab.RenderContext.create_default(
            shapelab.RenderConfig(640, 480))

        for idx in tqdm(range(0, self._max_frames, self.idx_skip), desc="Loading point-clouds"):

            snap = self.dataset[idx]
            points = snap.get_world_points()
            context.add_point_cloud(points, snap.colors/255)

        viewer = context.viewer()
        viewer.reset_view()
        viewer.wait_key()
