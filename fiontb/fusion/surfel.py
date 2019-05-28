import math
import time

import torch
import numpy as np

from fiontb.pointcloud import PointCloud, pcl_stack
import fiontb.fiontblib as fiontblib

from ._nearestpoints import KDTree
from fiontb.spatial import cOctree


def compute_surfel_radii(cam_points, normals, kcam):
    focal_len = abs(kcam.matrix[0, 0] + kcam.matrix[1, 1]) * .5
    radii = (
        cam_points[:, 2] / focal_len) * math.sqrt(2)
    radii = torch.min(2*radii, radii /
                      normals[:, 2].squeeze().abs())

    return radii


def compute_confidences(frame_points, kcam):
    camera_center = torch.tensor(kcam.pixel_center())

    xy_coords = frame_points[:, :2]

    confidences = torch.norm(
        xy_coords - camera_center, p=2, dim=1)
    confidences = confidences / confidences.max()

    confidences = torch.exp(-torch.pow(confidences, 2) /
                            (2.0*math.pow(0.6, 2)))

    return confidences


class SurfelData:
    def __init__(self, num_surfels, device):
        self.device = device
        self._max_surfel_count = num_surfels
        self.points = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.normals = torch.zeros(
            (num_surfels, 3), dtype=torch.float32).to(device)
        self.colors = torch.zeros(
            (num_surfels, 3), dtype=torch.uint8).to(device)
        self.radii = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.confs = torch.zeros(
            (num_surfels,), dtype=torch.float32).to(device)
        self.timestamps = torch.zeros(
            (num_surfels, ), dtype=torch.int32).to(device)
        self.surfel_mask = torch.ones(
            (num_surfels, ), dtype=torch.uint8).to(device)

    @classmethod
    def from_point_cloud(cls, frame_pcl, device, timestamp):
        cam_pcl = frame_pcl.unordered_point_cloud(world_space=False).torch()

        img_mask = frame_pcl.fg_mask.flatten()
        img_points = torch.from_numpy(
            frame_pcl.image_points.reshape(-1, 3)[img_mask])
        confs = compute_confidences(img_points, frame_pcl.kcam)

        radii = compute_surfel_radii(cam_pcl.points, cam_pcl.normals,
                                     frame_pcl.kcam)

        world_pcl = frame_pcl.unordered_point_cloud(world_space=True).torch()
        surfels = cls(cam_pcl.points.size(0), device)
        surfels.points[:] = world_pcl.points
        surfels.colors[:] = world_pcl.colors
        surfels.normals[:] = world_pcl.normals
        surfels.radii[:] = radii
        surfels.confs[:] = confs
        surfels.timestamps[:] = timestamp
        surfels.surfel_mask[:] = 0

        return surfels

    def mark_active(self, indices):
        self.surfel_mask[indices] = 0

    def mark_inactive(self, indices):
        self.surfel_mask[indices] = 1

    def num_active_surfels(self):
        return (self.surfel_mask == 0).sum()

    def get_active_indices(self):
        return (self.surfel_mask == 0).nonzero().flatten()

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

    def compact(self):
        active_mask = self.active_mask()

        compact = SurfelData(active_mask.sum(), self.device)
        compact.points[:] = self.points[active_mask]
        compact.normals[:] = self.normals[active_mask]
        compact.colors[:] = self.colors[active_mask]
        compact.radii = self.radii[active_mask]
        compact.confs = self.confs[active_mask]
        compact.timestamps = self.timestamps[active_mask]
        compact.surfel_mask[:] = 0
        return compact

    def share_memory(self):
        self.points = self.points.share_memory_()
        self.colors = self.colors.share_memory_()
        self.normals = self.normals.share_memory_()
        self.radii = self.radii.share_memory_()
        self.confs = self.confs.share_memory_()
        self.timestamps = self.timestamps.share_memory_()
        self.surfel_mask = self.surfel_mask.share_memory_()

    @property
    def max_surfel_count(self):
        return self._max_surfel_count

    def __str__(self):
        return "Surfel with {} points".format(self.num_active_surfels())

    def __repr__(self):
        return str(self)


def torch_pcl(pcl, device):
    pcl = PointCloud(torch.from_numpy(pcl.points).to(device),
                     torch.from_numpy(pcl.colors).to(device),
                     torch.from_numpy(pcl.normals).to(device))

    return pcl


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 stable_conf_thresh=10, max_unstable_time=15):
        self.surfels = surfels
        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0

        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True

        self._start_timestamp = 0
        self.merge_min_radio = 0.5

    def fuse(self, frame_pcl, live_pcl, kcam):
        img_mask = frame_pcl.fg_mask.flatten()

        # Computes confidences
        live_confidence = compute_confidences(
            torch.from_numpy(frame_pcl.points.reshape(-1, 3)[img_mask]),
            kcam).to(self.surfels.device)

        # Computes radii
        cam_pcl = frame_pcl.unordered_point_cloud(world_space=False).torch()
        live_radii = compute_surfel_radii(
            cam_pcl.points, cam_pcl.normals, kcam).to(self.surfels.device)

        # timestamp = int((time.time() - self._start_timestamp)*1000)
        timestamp = self._start_timestamp
        self._start_timestamp += 1

        live_pcl = live_pcl.torch(self.surfels.device)
        if self._is_first_fusion:
            new_idxs = torch.arange(0, live_pcl.size)

            self._add_surfels(new_idxs, new_idxs, live_pcl,
                              live_confidence, live_radii, timestamp)
            self._is_first_fusion = False
            return (new_idxs.to(self.surfels.device),
                    torch.tensor([], dtype=torch.int64))

        active_mask = self.surfels.active_mask()

        # nn_search = cOctree(self.surfels.points[active_mask], 2048)
        nn_search = KDTree(
            self.surfels.points[active_mask].cpu(), self.surfels.device)

        dist_mtx, idx_mtx = nn_search.query(live_pcl.points, 8, 0.01)

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

        model_remove_idxs = self._remove_surfels(active_mask)

        return model_update_idxs, model_remove_idxs

    def _fuse_surfels(self, live_pcl, live_confidence, live_radii, live_idxs, model_idxs):
        count = self.surfels.confs[model_idxs]
        count_update = count + live_confidence.squeeze()

        # update all visible surfels
        self.surfels.confs[model_idxs] = count_update

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
        self.surfels.confs[model_empty_idxs] = live_confidence[live_idxs]
        self.surfels.timestamps[model_empty_idxs] = timestamp

    def _remove_surfels(self, active_mask):
        # model_remove_idxs = torch.tensor([], dtype=torch.int64)
        unstable_idxs = (
            self.surfels.confs[active_mask] < self.stable_conf_thresh).nonzero().squeeze()

        remove_mask = (
            self.surfels.timestamps[unstable_idxs] >= self.max_unstable_time)
        remove_idxs = unstable_idxs[remove_mask]
        self.surfels.mark_inactive(remove_idxs)

        return remove_idxs

    def _merge_points(self, active_mask):
        stable_idxs = (
            self.surfels.count[active_mask] > self.max_unstable_count).nonzero().squeeze()
        nn_search = Search(self.surfels.points[active_mask])
        dist_mtx, idx_mtx = nn_search.search(self.surfels.points[stable_idxs])

        remove_idxs = set()
        for stable_idx, (dist, idx) in zip(stable_idxs, dist_mtx, idx_mtx):
            if dist < 0.01:
                remove_idxs.add(idx)
                self.sta


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
