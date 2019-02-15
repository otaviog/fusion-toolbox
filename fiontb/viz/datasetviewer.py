"""Dataset viewer
"""
import argparse
from collections import deque

import cv2
import numpy as np
from matplotlib.pyplot import get_cmap
from tqdm import tqdm

import shapelab
import shapelab.io


class _Viewer:
    def __init__(self, width, height):
        self.ctx = shapelab.RenderContext(width, height)
        self.view = self.ctx.viewer()


_CAM_MATRIX = np.eye(4)
_CAM_MATRIX[2, 2] = -1


def _gram_schmidt_columns(X):
    # pylint: disable=invalid-name
    import ipdb
    ipdb.set_trace()

    Q, _ = np.linalg.qr(X)
    return Q


class DatasetViewer:
    def __init__(self, dataset, title="Dataset"):
        self.dataset = dataset
        self.title = title
        self.show_mask = False
        self.last_proc_data = {'idx': -1}

        self.viewer_cam = _Viewer(640, 480)
        self.viewer_cam.view.set_title("{}: camera space".format(title))

        self.viewer_world = _Viewer(640, 480)
        self.viewer_world.view.set_title("{}: world space".format(title))
        self.visited_idxs = set()

        self.pcl_deque = deque()
        self.visited_idxs = set()

    def _update_world(self, idx, snap, cam_space, cam_proj):
        if idx in self.visited_idxs:
            return None

        rt_cam = snap.rt_cam.matrix
        world_space = np.matmul(
            np.matmul(_CAM_MATRIX, rt_cam), cam_space)
        pcl = self.viewer_world.ctx.add_point_cloud(
            world_space[:, 0:3], snap.colors)

        cam = self.viewer_world.ctx.add_camera(
            cam_proj, np.matmul(_CAM_MATRIX, rt_cam))

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

            self.viewer_cam.ctx.clear_scene()

            cam_space = snap.get_cam_points()
            cam_space = np.insert(cam_space, 3, 1.0, axis=1)

            self.viewer_cam.ctx.add_point_cloud(
                np.matmul(_CAM_MATRIX, cam_space)[:, 0:3], snap.colors)

            cam_proj = shapelab.projection_from_kcam(
                snap.kcam.matrix, 0.5, cam_space[:, 2].max())

            self.viewer_cam.view.reset_view()
            self.viewer_cam.view.set_view(0, 0)

            if snap.rt_cam is not None:
                self._update_world(idx, snap, cam_space, cam_proj)

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
            key = cv2.waitKey(1)
            self.viewer_cam.view.draw(0)
            self.viewer_world.view.draw(0)

            if key == 27:
                break

            key = chr(key & 0xff).lower()

            if key == 'q':
                break
            elif key == 'm':
                self.show_mask = not self.show_mask


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

        point_list = []
        color_list = []
        nodes = []
        for idx in tqdm(range(0, self._max_frames, self.idx_skip),
                        desc="Loading point-clouds"):
            snap = self.dataset[idx]
            points = snap.get_world_points().astype(np.float32)

            context.add_point_cloud(points, snap.colors/255)

        viewer = context.viewer()
        viewer.reset_view()
        viewer.wait_key()


def _main():
    from fiontb.data.klg import KLG
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_type", metavar='dataset-type',
                        choices=['klg'], help="Input klg file")
    parser.add_argument(
        "inputs", nargs='+',
        help="Input list, like base path and trajectory files, dependes on the dataset type.")

    args = parser.parse_args()

    if args.dataset_type == "klg":
        dataset = KLG(args.inputs[0])

    viewer = DatasetViewer(dataset)
    viewer.run()


if __name__ == '__main__':
    _main()
