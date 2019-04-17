import torch
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

import shapelab

from fiontb.datatypes import PointCloud, pcl_stack
from ._brutesearch import BruteNNSearch


class SceneSurfelData:
    def __init__(self, num_surfels, device):
        self.device = device
        self.points = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.normals = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.colors = torch.zeros(
            (num_surfels, 3), dtype=torch.int16).to(device)
        self.radius = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.counts = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.timestamps = torch.zeros(
            (num_surfels, ), dtype=torch.int32).to(device)
        self.surfel_mask = torch.ones(
            (num_surfels, ), dtype=torch.uint8).to(device)

    def mark_active(self, indices):
        self.surfel_mask[indices] = 0

    def get_unactive_indices(self, num):
        return self.surfel_mask.nonzero()[:num].flatten()

    def active_points(self):
        return self.points[self.active_mask()]

    def active_mask(self):
        return self.surfel_mask == 0

    def to_point_cloud(self):
        active_mask = self.active_mask()

        return PointCloud(
            points=self.points[active_mask].cpu().numpy(),
            colors=self.colors[active_mask].cpu().numpy().astype(np.uint8),
            normals=self.normals[active_mask].cpu().numpy())


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0):
        self.surfels = surfels
        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0
        self.first_fusion = True

    def fuse(self, live_pcl, camera_center):
        if self.first_fusion:
            indices = torch.arange(0, live_pcl.size)
            self.surfels.points[indices] = torch.from_numpy(
                live_pcl.points).to(self.surfels.device)
            self.surfels.colors[indices] = torch.from_numpy(
                live_pcl.colors).short().to(self.surfels.device)
            self.surfels.normals[indices] = torch.from_numpy(
                live_pcl.normals).to(self.surfels.device)

            self.surfels.mark_active(indices)
            self.first_fusion = False
            return indices.to(self.surfels.device)

        search = cKDTree(
            self.surfels.active_points().cpu(), balanced_tree=False)
        dist_mtx, idx_mtx = search.query(live_pcl.points, 1)
        dist_mtx = dist_mtx.flatten()
        idx_mtx = idx_mtx.flatten()

        live_pcl = PointCloud(torch.from_numpy(live_pcl.points).to(self.surfels.device),
                              torch.from_numpy(
                                  live_pcl.colors).short().to(self.surfels.device),
                              torch.from_numpy(live_pcl.normals).to(self.surfels.device))

        live_fuse_idxs = np.where(dist_mtx < self.max_distance)[0]
        model_fuse_idxs = idx_mtx[live_fuse_idxs]
        live_fuse_idxs = torch.from_numpy(live_fuse_idxs)
        model_fuse_idxs = torch.from_numpy(model_fuse_idxs)

        norm_dot = torch.bmm(
            live_pcl.normals[live_fuse_idxs].view(-1, 1, 3),
            self.surfels.normals[model_fuse_idxs].view(-1, 3, 1)).squeeze()

        good_norm_mask = (norm_dot > self.normal_min_dot)
        live_fuse_idxs = live_fuse_idxs[good_norm_mask]
        model_fuse_idxs = model_fuse_idxs[good_norm_mask]

        live_confidence = torch.norm(live_pcl.points - camera_center, p=4)
        live_confidence = torch.exp(-live_confidence/(2*0.6 ^ 2))

        # live_weights
        count = self.surfels.counts[model_fuse_idxs]
        count_update = count + live_confidence

        self.surfels.points[model_fuse_idxs] = (
            (self.surfels.points[model_fuse_idxs]*count + live_pcl.points[live_fuse_idxs] * live_confidence) /
            count_update)

        self.surfels.points[model_fuse_idxs] += live_pcl.points[live_fuse_idxs]
        self.surfels.points[model_fuse_idxs] *= 0.5

        self.surfels.colors[model_fuse_idxs] += live_pcl.colors[live_fuse_idxs]
        self.surfels.colors[model_fuse_idxs] /= 2

        self.surfels.normals[model_fuse_idxs] += live_pcl.normals[live_fuse_idxs]
        self.surfels.normals[model_fuse_idxs] *= 0.5
        self.surfels.normals[model_fuse_idxs] /= torch.norm(
            self.surfels.normals[model_fuse_idxs], dim=0)

        self.surfels.counts[model_fuse_idxs] = count_update

        live_new_idxs = torch.from_numpy(
            np.where(dist_mtx > self.max_distance)[0])
        live_new_idxs = live_new_idxs.flatten()

        empty_indices = self.surfels.get_unactive_indices(
            live_new_idxs.shape[0])
        self.surfels.mark_active(empty_indices)

        self.surfels.points[empty_indices] = live_pcl.points[live_new_idxs]
        self.surfels.colors[empty_indices] = live_pcl.colors[live_new_idxs]
        self.surfels.normals[empty_indices] = live_pcl.normals[live_new_idxs]

        update_indices = torch.cat(
            [model_fuse_idxs.to(self.surfels.device), empty_indices])
        return update_indices


class DenseFusion:
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
