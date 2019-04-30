import math
import time

import torch
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree
import cv2

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
        self.radii = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.counts = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.timestamps = torch.zeros(
            (num_surfels, ), dtype=torch.int32).to(device)
        self.surfel_mask = torch.ones(
            (num_surfels, ), dtype=torch.uint8).to(device)

    def mark_active(self, indices):
        self.surfel_mask[indices] = 0

    def mark_inactive(self, indices):
        self.surfel_mask[indices] = 1

    def get_inactive_indices(self, num):
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

    def share_memory(self):
        self.points = self.points.share_memory_()
        self.colors = self.colors.share_memory_()
        self.normals = self.normals.share_memory_()
        self.radii = self.radii.share_memory_()
        self.counts = self.counts.share_memory_()
        self.timestamps = self.timestamps.share_memory_()
        self.surfel_mask = self.surfel_mask.share_memory_()


def torch_pcl(pcl, device):
    pcl = PointCloud(torch.from_numpy(pcl.points).to(device),
                     torch.from_numpy(pcl.colors).short().to(device),
                     torch.from_numpy(pcl.normals).to(device))

    return pcl


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 max_unstable_time=10000):
        self.surfels = surfels
        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True
        self._start_timestamp = time.time()

    def _get_confidence(self, frame_points):
        camera_center = torch.tensor(frame_points.kcam.pixel_center())

        xy = torch.from_numpy(frame_points.points[:, :2]).squeeze()

        confidence = torch.norm(
            xy - camera_center, p=2, dim=1)
        confidence = confidence / confidence.max()
        
        confidence = torch.exp(-torch.pow(confidence, 2) /
                               (2.0*math.pow(0.6, 2)))

        return confidence.to(self.surfels.device)

    def fuse(self, frame_points, live_pcl, kcam, rt_cam):
        focal_len = (kcam.matrix[0, 0] + kcam.matrix[1, 1]) * .5

        timestamp = int((time.time() - self._start_timestamp)*1000)
        if self._is_first_fusion:
            indices = torch.arange(0, live_pcl.size)
            live_pcl = torch_pcl(live_pcl, self.surfels.device)
            live_confidence = self._get_confidence(frame_points)

            self._add_surfels(indices, indices, live_pcl,
                              live_confidence, timestamp, focal_len)
            self._is_first_fusion = False
            return (indices.to(self.surfels.device),
                    torch.tensor([], dtype=torch.int64))

        active_mask = self.surfels.active_mask()

        search = cKDTree(
            self.surfels.points[active_mask].cpu(), balanced_tree=False)
        dist_mtx, idx_mtx = search.query(live_pcl.points, 1)
        dist_mtx = dist_mtx.flatten()
        idx_mtx = idx_mtx.flatten()

        live_pcl = torch_pcl(live_pcl, self.surfels.device)

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

        live_confidence = self._get_confidence(frame_points)

        self._fuse_surfels(live_confidence[live_fuse_idxs], live_pcl,
                           model_fuse_idxs, live_fuse_idxs)

        live_new_idxs = torch.from_numpy(
            np.where(dist_mtx > self.max_distance)[0])
        live_new_idxs = live_new_idxs.flatten()

        empty_indices = self.surfels.get_inactive_indices(
            live_new_idxs.shape[0])
        if empty_indices.size(0) > 0:
            self._add_surfels(empty_indices, live_new_idxs, live_pcl,
                              live_confidence, timestamp, focal_len)

        update_indices = torch.cat(
            [model_fuse_idxs.to(self.surfels.device), empty_indices])

        self.surfels.timestamps[update_indices] = timestamp

        remove_indices = self._remove_surfels(timestamp, active_mask)

        return update_indices, remove_indices

    def _fuse_surfels(self, live_confidence, live_pcl, model_fuse_idxs, live_fuse_idxs):
        count = self.surfels.counts[model_fuse_idxs]
        count_update = count + live_confidence

        # For element-wise multiplication
        count = count.view(-1, 1)
        count_update = count_update.view(-1, 1)
        live_confidence = live_confidence.view(-1, 1)

        model_point_update = self.surfels.points[model_fuse_idxs] * count
        live_points = live_pcl.points[live_fuse_idxs]*live_confidence
        self.surfels.points[model_fuse_idxs] = (
            model_point_update + live_points) / count_update

        self.surfels.colors[model_fuse_idxs] += live_pcl.colors[live_fuse_idxs]
        self.surfels.colors[model_fuse_idxs] /= 2

        model_normal_update = self.surfels.normals[model_fuse_idxs]*count
        live_normals = live_pcl.normals[live_fuse_idxs]*live_confidence
        self.surfels.normals[model_fuse_idxs] = (
            model_normal_update + live_normals) / count_update

        self.surfels.counts[model_fuse_idxs] = count_update.squeeze()

    def _add_surfels(self, empty_indices, live_indices, live_pcl, live_confidence,
                     timestamp, focal_len):

        self.surfels.mark_active(empty_indices)

        points = live_pcl.points[live_indices]
        self.surfels.points[empty_indices] = points
        self.surfels.colors[empty_indices] = live_pcl.colors[live_indices]
        normals = live_pcl.normals[live_indices]
        self.surfels.normals[empty_indices] = normals

        radii = (points[:, 2] / focal_len) * math.sqrt(2)
        self.surfels.radii[empty_indices] = torch.min(
            2*radii, radii/normals[:, 2].abs())

        self.surfels.counts[empty_indices] = live_confidence[live_indices]
        self.surfels.timestamps[empty_indices] = timestamp

    def _remove_surfels(self, curr_timestamp, active_mask):
        return torch.tensor([], dtype=torch.int64)
        remove_mask = (
            curr_timestamp - self.surfels.timestamps[active_mask]) > self.max_unstable_time
        remove_indices = active_mask.nonzero().squeeze()[
            remove_mask].nonzero().view(-1)
        self.surfels.mark_inactive(remove_indices)

        return remove_indices


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
