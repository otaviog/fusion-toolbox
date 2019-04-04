import numpy as np
from sklearn.neighbors import KDTree

import shapelab

from fiontb.datatypes import PointCloud, pcl_stack


class SurfelFusion:
    def __init__(self):
        self.current_pcl = PointCloud()

    def fuse(self, live_pcl):
        if self.current_pcl.is_empty():
            self.current_pcl = live_pcl
            return

        tree = KDTree(self.current_pcl.points.squeeze())

        dist_mtx, idx_mtx = tree.query(live_pcl.points.squeeze(), 1)

        points = []
        colors = []
        normals = []

        for live_idx, (dist_row, idx_row) in enumerate(zip(dist_mtx, idx_mtx)):
            dist = dist_row[0]
            idx = idx_row[0]
            live_pos = live_pcl.points[live_idx]
            live_norm = live_pcl.normals[live_idx]
            live_color = live_pcl.colors[live_idx]

            if dist > 0.1:
                points.append(live_pos)
                colors.append(live_color)
                normals.append(live_norm)
                continue

            model_norm = self.current_pcl.normals[idx]
            if np.dot(live_norm, model_norm) > 0.5:
                continue

            model_pos = self.current_pcl.points[idx]
            new_pos = (live_pos + model_pos)*.5
            self.current_pcl.points[idx] = new_pos

            new_norm = (live_norm + model_norm)*0.5
            self.current_pcl.normals[idx] = new_norm

            model_color = self.current_pcl.colors[idx]
            new_color = (live_color + model_color)*.5
            self.current_pcl.colors[idx] = new_color

        new_points = PointCloud(np.array(points), np.array(colors), np.array(normals))
        self.current_pcl = pcl_stack([self.current_pcl, new_points])
            
    def get_model(self):
        return self.current_pcl

    def get_odometry_model(self):
        return self.current_pcl


class SurfelView:
    def __init__(self):
        self.ctx = shapelab.RenderContext(640, 480)
        self.viewer = self.ctx.viewer()

    def update(self, model):
        self.ctx.clear_scene()
        self.ctx.add_point_cloud(model.points, model.colors)


class DensePCLFusion:
    def __init__(self, keep_frames, sample_size):
        self.pcls = []
        self.keep_frames = keep_frames
        self.sample_size = sample_size
        self.reduced_set = set()

    def fuse(self, live_pcl):
        self.pcls.append(live_pcl)

        if len(self.pcls) < self.keep_frames:
            return

        for i, pcl in enumerate(self.pcls[:-self.keep_frames]):
            if i in self.reduced_set:
                continue

            which_points = np.random.choice(
                pcl.points.shape[0], int(pcl.points.shape[0]*self.sample_size),
                replace=False)

            pcl.points = pcl.points[which_points, :]
            pcl.colors = pcl.colors[which_points, :]
            pcl.normals = pcl.normals[which_points, :]

            self.reduced_set.add(i)

    def get_model(self):
        if not self.pcls:
            return PointCloud()
        return pcl_stack(self.pcls)

    def get_odometry_model(self):
        if not self.pcls:
            return PointCloud()

        return pcl_stack(self.pcls[-self.keep_frames:])
