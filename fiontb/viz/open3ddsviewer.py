"""Viewer using Open3D module
"""

from tqdm import tqdm
from open3d import PointCloud, draw_geometries

import numpy as np

class DatasetViewer:
    def __init__(self, dataset, idx_skip=1, max_frames=None):
        self.dataset = dataset
        self.idx_skip = idx_skip
        if max_frames is not None:
            self._max_frames = min(max_frames, len(dataset))
        else:
            self._max_frames = len(dataset)

    def run(self):
        pcl = PointCloud()
        for idx in tqdm(range(0, self._max_frames, self.idx_skip), desc="Loading point-clouds"):
            snap = self.dataset[idx]
            points = snap.get_world_points()

            for point, color in zip(points, snap.colors):
                pcl.points.append(point.squeeze().astype(np.float32))
                pcl.colors.append(color/255)
        draw_geometries([pcl])

