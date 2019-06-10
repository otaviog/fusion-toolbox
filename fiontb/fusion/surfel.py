import math
import time

import torch
import numpy as np
import tenviz

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


class SurfelCloud:
    @staticmethod
    def from_frame_pcl(cls, frame_pcl):
        cam_pcl = frame_pcl.unordered_point_cloud(world_space=False).torch()

        img_mask = frame_pcl.fg_mask.flatten()
        img_points = torch.from_numpy(
            frame_pcl.image_points.reshape(-1, 3)[img_mask])
        img_mask = frame_pcl.fg_mask.flatten()
        confs = compute_confidences(img_points, frame_pcl.kcam)

        radii = compute_surfel_radii(cam_pcl.points, cam_pcl.normals,
                                     frame_pcl.kcam)

        return cls(cam_pcl.points, cam_pcl.colors, cam_pcl.normals, radii, confs):

    def __init__(self, points, colors, normals, radii, confs):
        self.points = points
        self.colors = colors
        self.normals = normals
        self.radii = radii
        self.confs = confs

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


class SurfelsModel:
    def __init__(self, context, max_surfels):
        self.context = context
        self._max_surfels = max_surfels

        with self.context.current():
            self.points = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.points_tex = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.normals = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.colors = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Uint8)
            self.radii = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.confs = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.timestamps = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Int32)
            self.surfel_mask = torch.ones((max_surfels, ), dtype=torch.uint8)

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

    @property
    def max_surfel_count(self):
        return self._max_surfel_count

    def __str__(self):
        return "Surfel with {} points".format(self.num_active_surfels())

    def __repr__(self):
        return str(self)


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 stable_conf_thresh=10, max_unstable_time=15):
        self.surfels = surfels
        self.indexmap = IndexMap(surfels.context, surfels)

        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0

        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True

        self._start_timestamp = 0
        self.merge_min_radio = 0.5

    def fuse(self, frame_pcl, live_pcl, kcam):
        surfel_cloud = SurfelCloud.from_frame_pcl(frame_pcl)

        timestamp = self._start_timestamp
        self._start_timestamp += 1

        live_pcl = live_pcl.torch(self.surfels.device)
        if self._is_first_fusion:
            new_idxs = torch.arange(0, live_pcl.size)

            self._add_surfels(new_idxs, new_idxs, live_pcl,
                              live_confidence, live_radii, timestamp)
            self._is_first_fusion = False
            return

        live_idxs, model_idxs, live_unst_idxs = self.indexmap.query(
            frame_pcl, surfel_cloud)

        if live_unst_idxs.size(0) > 0:

            self._add_surfels(live_surfels.indexed_select(live_unst_idxs))

        with self.surfels.context.current():
            model_surfels = SurfelCloud(
                self.surfels.points[model_idxs],
                self.surfels.colors[model_idxs],
                self.surfels.normals[model_idxs],
                self.surfels.radii[model_idxs],
                self.surfels.confs[model_idxs])

        self._merge_surfels(
            live_surfels.indexed_select(live_idxs),
            model_surfels, model_idxs)

        self.surfels.timestamps[model_update_idxs] = timestamp

        # self._remove_surfels(active_mask)

    def _merge_surfels(self, live, model, model_idxs):

        self.model_surfels.confs = confs_update

        radii_mask = (live.radii < model.radii *
                      (1.0 + self.merge_min_radio))
        # live_idxs = live_idxs[radii_mask]
        # model_idxs = model_idxs[radii_mask]

        confs = confs[radii_mask].view(-1, 1)
        confs_update = confs_update[radii_mask].view(-1, 1)

        live_radii = live_radii[radii_mask].view(-1, 1)

        confs_update = model.confs + live.confs.squeeze()
        model.points = (model.points * confs + live.points *
                        live.confs) / confs_update
        model.colors = (model.colors * confs + live.colors *
                        live.confs) / confs_update

        model.normals = (model.normals * confs +
                         live.normals*live.confs) / confs_update
        model.normals /= model.normals.norm(2, 1).view(-1, 1)
        model.confs = confs_update

        with self.surfels.context.current():
            self.surfels.points[model_idxs] = model.points
            self.surfels.colors[model_idxs] = model.colors
            self.surfels.normals[model_idxs] = model.normals
            self.surfels.confs[model_idxs] = model.confs

    def _add_surfels(self, new_surfels):

        new_indices = self.surfels.get_inactive_indices(
            new_surfels.size)
        self.surfels.mark_active(new_indices)

        with self.surfels.context.current():
            self.surfels.points[new_indices] = new_surfels.points
            self.surfels.colors[new_indices] = new_surfels.colors
            self.surfels.normals[new_indices] = new_surfels.normals
            self.surfels.radii[new_indices] = new_surfels.radii
            self.surfels.confs[new_indices] = new_surfels.confs
            # self.surfels.timestamps[new_indices] = timestamp

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
            self.surfels.confs[active_mask] > self.max_unstable_confs).nonzero().squeeze()
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
