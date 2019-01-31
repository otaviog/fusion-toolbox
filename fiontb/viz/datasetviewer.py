"""Dataset viewer
"""

import cv2
import numpy as np
from matplotlib.pyplot import get_cmap

import shapelab


class DatasetViewer:    
    def __init__(self, dataset, title="Dataset"):
        self.dataset = dataset
        self.title = title
        self.show_mask = False
        self.last_proc_data = {'idx': -1}
        
        self.viewer3d = shapelab.AsyncViewer.create_default(
            shapelab.RenderConfig(640, 480))

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

            self.viewer3d.clear_scene()
            self.viewer3d.add_point_cloud(snap.cam_points, snap.colors/255)
            self.viewer3d.reset_view()

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
            self.dataset), self._update_canvas)
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
        self.viewer3d.stop()
