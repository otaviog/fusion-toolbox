
import cv2
import numpy as np
from matplotlib.pyplot import get_cmap


class FrameUI:

    _DEPTH_OPPACITY_LABEL = "Depth oppacity"
    _NORMAL_OPPACITY_LABEL = "normal oppacity"

    def __init__(self, title="slamtb.ui.FrameUI"):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(FrameUI._DEPTH_OPPACITY_LABEL,
                           self.title, 50, 100,
                           self._update)
        cv2.createTrackbar(FrameUI._NORMAL_OPPACITY_LABEL,
                           self.title, 50, 100,
                           self._update)

        self.frame = None
        self.normal_image = None

    def _update(self, _):
        if self.frame is None:
            return

        cmap = get_cmap('plasma', int(self.frame.info.depth_max))
        depth_img = (self.frame.depth_image / self.frame.info.depth_max)
        depth_img = cmap(depth_img)
        depth_img = depth_img[:, :, 0:3]
        depth_img = (depth_img*255).astype(np.uint8)

        depth_alpha = cv2.getTrackbarPos(
            FrameUI._DEPTH_OPPACITY_LABEL, self.title) / 100.0

        canvas = cv2.addWeighted(depth_img, depth_alpha,
                                 self.frame.rgb_image, 1.0 - depth_alpha, 0.0)

        if self.normal_image is not None:
            normal_alpha = cv2.getTrackbarPos(
                FrameUI._NORMAL_OPPACITY_LABEL, self.title) / 100.0

            canvas = cv2.addWeighted(self.normal_image, normal_alpha,
                                     canvas, 1.0 - normal_alpha, 0.0)

        cv2.imshow(self.title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def update(self, frame):
        self.frame = frame
        if frame.normal_image is not None:
            self.normal_image = convert_normals_to_rgb(
                frame.normal_image)
        self._update(0)


def convert_normals_to_rgb(normal_image):
    normal_image = (normal_image + 1)*0.5*255
    return normal_image.astype(np.uint8)
