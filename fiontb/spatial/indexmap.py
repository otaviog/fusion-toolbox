import torch

import tenviz

from fiontb.camera import Homogeneous
from fiontb._cfiontb import raster_indexmap


class IndexMap:
    def __init__(self, width, height, device="cuda:0"):
        self.indexmap = torch.empty(
            height, width, dtype=torch.int64).to(device)
        self.depth_buffer = torch.empty(
            height, width, dtype=torch.int32).to(device)
        self.device = device

    def raster(self, points, kcam):
        points = points.to(self.device)

        proj_matrix = tenviz.projection_from_kcam(
            kcam.matrix, 0.01, 100.0).to_matrix()
        proj_matrix = torch.from_numpy(proj_matrix).float().to(self.device)

        self.indexmap[:] = -1
        self.depth_buffer[:] = -1
        raster_indexmap(points, proj_matrix, self.indexmap, self.depth_buffer)


def _test():
    from pathlib import Path

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    from fiontb.frame import Frame, FrameInfo, FramePointCloud
    from fiontb.camera import KCamera

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))

    test_data = Path(__file__).parent / "../fusion/surfel/_test"
    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH).astype(np.int32)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    pcl = FramePointCloud(frame).unordered_point_cloud(world_space=False)

    idxmap = IndexMap(640, 480)
    idxmap.raster(pcl.points, kcam)

    plt.figure()
    plt.imshow(idxmap.indexmap.cpu().numpy())
    plt.figure()
    plt.imshow(idxmap.depth_buffer.cpu().numpy())
    plt.show()


if __name__ == '__main__':
    _test()
