"""Surfel fusion. Based on the following papers:

* Keller, Maik, Damien Lefloch, Martin Lambers, Shahram Izadi, Tim
  Weyrich, and Andreas Kolb. "Real-time 3d reconstruction in dynamic
  scenes using point-based fusion." In 2013 International Conference
  on 3D Vision-3DV 2013, pp. 1-8. IEEE, 2013.

* Whelan, Thomas, Stefan Leutenegger, R. Salas-Moreno, Ben Glocker,
  and Andrew Davison. "ElasticFusion: Dense SLAM without a pose
  graph." Robotics: Science and Systems, 2015.

Based on the code of ElasticFusion:

* https://github.com/mp3guy/ElasticFusion

"""
import math

import torch
import tenviz

from fiontb.camera import Homogeneous, normal_transform_matrix

from .indexmap import IndexMap


def compute_surfel_radii(cam_points, normals, kcam):
    focal_len = abs(kcam.matrix[0, 0] + kcam.matrix[1, 1]) * .5
    radii = (
        cam_points[:, 2] / focal_len) * math.sqrt(2)
    radii = torch.min(2*radii, radii /
                      normals[:, 2].squeeze().abs())

    return radii


def compute_confidences(frame_pcl, no_mask=False):
    img_points = frame_pcl.image_points[:, :, :2].reshape(-1, 2)
    img_mask = frame_pcl.fg_mask.flatten()
    
    if not no_mask:
        img_points = img_points[img_mask]

    img_points = torch.from_numpy(img_points)

    camera_center = torch.tensor(frame_pcl.kcam.pixel_center())

    confidences = torch.norm(
        img_points - camera_center, p=2, dim=1)
    confidences = confidences / confidences.max()

    confidences = torch.exp(-torch.pow(confidences, 2) /
                            (2.0*math.pow(0.6, 2)))

    return confidences


class _ConfidenceCache:
    def __init__(self):
        self.width = -1
        self.height = -1
        self.confidences = None

    def get_confidences(self, frame_pcl):
        fheight, fwidth = frame_pcl.image_points.shape[:2]
        # It doesn't check kcam
        if fheight != self.height or fwidth != self.width:
            self.width = fwidth
            self.height = fheight
            self.confidences = compute_confidences(frame_pcl, no_mask=True)

        return self.confidences[torch.from_numpy(frame_pcl.fg_mask.flatten())]


class SurfelCloud:
    """Compact surfels representation in PyTorch.
    """

    @classmethod
    def from_frame_pcl(cls, frame_pcl, time, device, confs=None):
        cam_pcl = frame_pcl.unordered_point_cloud(world_space=False).torch()

        if confs is None:
            confs = compute_confidences(frame_pcl)

        radii = compute_surfel_radii(cam_pcl.points, cam_pcl.normals,
                                     frame_pcl.kcam)
        times = torch.full((cam_pcl.points.size(0),), time,
                           dtype=torch.int32).to(device)
        return cls(cam_pcl.points.to(device), cam_pcl.colors.to(device), cam_pcl.normals.to(device),
                   radii.to(device), confs.to(device), times, device)

    def __init__(self, points, colors, normals, radii, confs, times, device):
        self.points = points
        self.colors = colors
        self.normals = normals
        self.radii = radii
        self.confs = confs
        self.times = times
        self.device = device

    def to(self, device):
        self.points = self.points.to(device)
        self.colors = self.colors.to(device)
        self.normals = self.normals.to(device)
        self.radii = self.radii.to(device)
        self.confs = self.confs.to(device)
        self.times = self.times.to(device)
        self.device = device

    def index_select(self, index):
        return SurfelCloud(self.points[index],
                           self.colors[index],
                           self.normals[index],
                           self.radii[index],
                           self.confs[index],
                           self.times[index],
                           self.device)

    def transform(self, matrix):
        if self.points.size(0) == 0:
            return

        self.points = Homogeneous(torch.from_numpy(
            matrix).to(self.device)) @ self.points
        normal_matrix = torch.from_numpy(
            normal_transform_matrix(matrix)).to(self.device)
        self.normals = (
            normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    @property
    def size(self):
        return self.points.size(0)


class SurfelModel:
    """Global surfel model on GL buffers.
    """

    def __init__(self, context, max_surfels, device="cuda:0"):
        self.context = context
        self.max_surfels = max_surfels
        self.device = device

        self.active_mask = torch.ones(
            (max_surfels, ), dtype=torch.uint8).to(device)

        with self.context.current():
            self.points = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.points_tex = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.normals = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.colors = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Uint8, normalize=True)
            self.radii = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.confs = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.times = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Int32)
            self.active_mask_gl = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Uint8, integer_attrib=True)
            self.active_mask_gl.from_tensor(self.active_mask)
        self.max_time = 0

    def mark_active(self, indices):
        self.active_mask[indices] = 0

    def mark_inactive(self, indices):
        self.active_mask[indices] = 1

    def get_active_indices(self):
        return (self.active_mask == 0).nonzero().flatten()

    def get_inactive_indices(self, num):
        return self.active_mask.nonzero()[:num].flatten()

    def active_mask(self):
        return self.active_mask == 0

    def num_active_surfels(self):
        return (self.active_mask == 0).sum()

    def update_active_mask_gl(self):
        with self.context.current():
            self.active_mask_gl.from_tensor(self.active_mask.contiguous())

    def add_surfels(self, new_surfels):
        new_indices = self.get_inactive_indices(
            new_surfels.size)
        self.mark_active(new_indices)

        with self.context.current():
            self.points[new_indices] = new_surfels.points
            self.colors[new_indices] = new_surfels.colors
            self.normals[new_indices] = new_surfels.normals
            self.radii[new_indices] = new_surfels.radii
            self.confs[new_indices] = new_surfels.confs
            self.times[new_indices] = new_surfels.times

    def __str__(self):
        return "SurfelModel with {} active points, {} max. capacity".format(
            self.num_active_surfels(), self.max_surfels)

    def __repr__(self):
        return str(self)


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 stable_conf_thresh=10, max_unstable_time=20):
        self.surfels = surfels
        self.indexmap = IndexMap(surfels.context, surfels)

        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0

        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True

        self._time = 0
        self.merge_min_radio = 0.5

        self._conf_compute_cache = _ConfidenceCache()

    def fuse(self, frame_pcl, kcam, rt_cam):
        device = "cuda:0"

        frame_confs = self._conf_compute_cache.get_confidences(frame_pcl)
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl, self._time, device, confs=frame_confs)

        if self._is_first_fusion:
            live_surfels.transform(rt_cam.cam_to_world)
            self.surfels.add_surfels(live_surfels)
            self.surfels.update_active_mask_gl()
            self._is_first_fusion = False
            return

        height, width = frame_pcl.image_points.shape[:2]

        indexmap_scale = 1
        self.indexmap.update(rt_cam, int(width*indexmap_scale),
                             int(height*indexmap_scale),
                             kcam, 0.01, 10.0)

        live_idxs, model_idxs, live_unst_idxs = self.indexmap.query(
            live_surfels, width, height, kcam, 0.01, 10.0)
        model_idxs = model_idxs.to(device)

        live_surfels.transform(rt_cam.cam_to_world)
        if live_unst_idxs.size(0) > 0:
            self.surfels.add_surfels(live_surfels.index_select(live_unst_idxs))

        with self.surfels.context.current():
            model_surfels = SurfelCloud(
                self.surfels.points[model_idxs],
                self.surfels.colors[model_idxs],
                self.surfels.normals[model_idxs],
                self.surfels.radii[model_idxs].squeeze(),
                self.surfels.confs[model_idxs].squeeze(),
                self.surfels.times[model_idxs].squeeze(),
                device)

        live_idxs = live_idxs.to(device)
        self._merge_surfels(
            live_surfels.index_select(live_idxs),
            model_surfels, model_idxs.to(device))

        # visible_model_idxs = self.indexmap.get_visible_model_indices().to(device)

        # self._remove_surfels(visible_model_idxs)

        self._time += 1
        self.surfels.max_time = self._time

    def _merge_surfels(self, live, model, model_idxs):
        # radii_mask = (live.radii < model.radii *
        #              (1.0 + self.merge_min_radio))
        # live_idxs = live_idxs[radii_mask]
        # model_idxs = model_idxs[radii_mask]

        # confs = model.confs[radii_mask].view(-1, 1)
        # confs_update = confs_update[radii_mask].view(-1, 1)
        # live_radii = live.radii[radii_mask].view(-1, 1)

        confs_update = (model.confs + live.confs).view(-1, 1)
        model.points = (model.points * model.confs.view(-1, 1) + live.points *
                        live.confs.view(-1, 1)) / confs_update
        model.colors = (model.colors.float() * model.confs.view(-1, 1) + live.colors.float() *
                        live.confs.view(-1, 1)) / confs_update
        model.colors = model.colors.byte()

        model.normals = (model.normals * model.confs.view(-1, 1) +
                         live.normals*live.confs.view(-1, 1)) / confs_update
        model.normals /= model.normals.norm(2, 1).view(-1, 1)
        model.confs = confs_update.squeeze()

        with self.surfels.context.current():
            self.surfels.points[model_idxs] = model.points
            self.surfels.colors[model_idxs] = model.colors
            self.surfels.normals[model_idxs] = model.normals
            self.surfels.confs[model_idxs] = model.confs

    def _remove_surfels(self, visible_model_idxs):
        with self.surfels.context.current():
            confs = self.surfels.confs[visible_model_idxs].squeeze()
            times = self.surfels.times[visible_model_idxs].squeeze()

        unstable_idxs = visible_model_idxs[(confs < self.stable_conf_thresh)
                                           & (self._time - times >= self.max_unstable_time)]

        self.surfels.mark_inactive(unstable_idxs)
