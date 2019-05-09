import math
import time

import torch
import numpy as np

from fiontb.datatypes import PointCloud, pcl_stack
import fiontb.fiontblib as fiontblib

from ._nearestpoints import KDTree


class SceneSurfelData:
    def __init__(self, num_surfels, device):
        self.device = device
        self.points = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.normals = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.colors = torch.zeros(
            (num_surfels, 3), dtype=torch.uint8).to(device)
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
                     torch.from_numpy(pcl.colors).to(device),
                     torch.from_numpy(pcl.normals).to(device))

    return pcl


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 max_unstable_time=4):
        self.surfels = surfels
        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True
        # self._start_timestamp = time.time()
        self._start_timestamp = 0
        self.merge_min_radio = 0.5

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
        live_pcl = torch_pcl(live_pcl, self.surfels.device)

        focal_len = (kcam.matrix[0, 0] + kcam.matrix[1, 1]) * .5
        live_confidence = self._get_confidence(frame_points)

        # Computes radii
        live_radii = (
            live_pcl.points[:, 2].squeeze() / focal_len) * math.sqrt(2)
        live_radii = torch.min(2*live_radii, live_radii /
                               live_pcl.normals[:, 2].squeeze().abs())

        # timestamp = int((time.time() - self._start_timestamp)*1000)
        timestamp = self._start_timestamp
        self._start_timestamp += 1
        if self._is_first_fusion:
            new_idxs = torch.arange(0, live_pcl.size)

            self._add_surfels(new_idxs, new_idxs, live_pcl,
                              live_confidence, live_radii, timestamp)
            self._is_first_fusion = False
            return (new_idxs.to(self.surfels.device),
                    torch.tensor([], dtype=torch.int64))

        active_mask = self.surfels.active_mask()
        nn_search = KDTree(
            self.surfels.points[active_mask].cpu(), self.surfels.device)
        dist_mtx, idx_mtx = nn_search.query(live_pcl.points)
        
        fuse_idxs = fiontblib.filter_search(
            dist_mtx.cpu(), idx_mtx.cpu(), live_pcl.normals.cpu(),
            self.surfels.normals[active_mask].cpu(), 0.05,
            self.normal_min_dot)

        fuse_idxs = fuse_idxs.to(self.surfels.device)
        live_fuse_idxs = (fuse_idxs >= 0).nonzero().squeeze()
        model_fuse_idxs = fuse_idxs[live_fuse_idxs]
        model_fuse_idxs = active_mask.nonzero().squeeze()[
            model_fuse_idxs].squeeze()

        model_update_idxs = self._fuse_surfels(
            live_pcl, live_confidence[live_fuse_idxs], live_radii[live_fuse_idxs],
            live_fuse_idxs, model_fuse_idxs)
        self.surfels.timestamps[model_update_idxs] = timestamp

        live_new_idxs = (fuse_idxs == -1).nonzero().squeeze()
        if live_new_idxs.size(0) > 0:
            new_indices = self.surfels.get_inactive_indices(
                live_new_idxs.shape[0])

            self._add_surfels(new_indices, live_new_idxs, live_pcl,
                              live_confidence, live_radii, timestamp)
            model_update_idxs = torch.cat([model_update_idxs, new_indices])
        
        # model_remove_idxs = self._remove_surfels(timestamp, active_mask)

        model_remove_idxs = torch.tensor([], dtype=torch.int64)
        return model_update_idxs, model_remove_idxs

    def _fuse_surfels(self, live_pcl, live_confidence, live_radii, live_idxs, model_idxs):
        count = self.surfels.counts[model_idxs]
        count_update = count + live_confidence.squeeze()

        # update all visible surfels
        self.surfels.counts[model_idxs] = count_update

        radii_mask = (live_radii < self.surfels.radii[model_idxs] *
                      (1.0 + self.merge_min_radio))
        live_idxs = live_idxs[radii_mask]
        model_idxs = model_idxs[radii_mask]

        count = count[radii_mask].view(-1, 1)
        count_update = count_update[radii_mask].view(-1, 1)

        live_confidence = live_confidence[radii_mask].view(-1, 1)
        live_radii = live_radii[radii_mask].view(-1, 1)

        # point update
        model_point_update = self.surfels.points[model_idxs] * count
        live_points = live_pcl.points[live_idxs]*live_confidence
        self.surfels.points[model_idxs] = (
            model_point_update + live_points) / count_update

        # color update
        model_color_update = (self.surfels.colors[model_idxs].float()
                              * count)
        live_color = live_pcl.colors[live_idxs].float()*live_confidence
        self.surfels.colors[model_idxs] = ((
            model_color_update + live_color) / count_update).byte()

        # normal update
        model_normal_update = self.surfels.normals[model_idxs]*count
        live_normals = live_pcl.normals[live_idxs]*live_confidence
        normals = model_normal_update + live_normals
        normals /= count_update
        normals /= normals.norm(2, 1).view(-1, 1)
        self.surfels.normals[model_idxs] = normals

        return model_idxs

    def _add_surfels(self, model_empty_idxs, live_idxs, live_pcl, live_confidence,
                     live_radii, timestamp):

        self.surfels.mark_active(model_empty_idxs)

        self.surfels.points[model_empty_idxs] = live_pcl.points[live_idxs]
        self.surfels.colors[model_empty_idxs] = live_pcl.colors[live_idxs]
        self.surfels.normals[model_empty_idxs] = live_pcl.normals[live_idxs]
        self.surfels.radii[model_empty_idxs] = live_radii[live_idxs]
        self.surfels.counts[model_empty_idxs] = live_confidence[live_idxs]
        self.surfels.timestamps[model_empty_idxs] = timestamp

    def _remove_surfels(self, curr_timestamp, active_mask):
        unstable_idxs = (self.surfels.count[active_mask] < self.max_unstable_count).nonzero().squeeze()
        #unstable_idxs
        
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
