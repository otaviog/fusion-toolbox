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


def compute_surfel_radii(cam_points, normals, kcam):
    focal_len = abs(kcam.matrix[0, 0] + kcam.matrix[1, 1]) * .5
    radii = (
        cam_points[:, 2] / focal_len) * math.sqrt(2)
    radii = torch.min(2*radii, radii /
                      normals[:, 2].squeeze().abs())

    return radii


def compute_confidences(frame_pcl, no_mask=False):
    img_points = frame_pcl.image_points[:, :, :2].reshape(-1, 2)
    img_mask = frame_pcl.mask.flatten()

    if not no_mask:
        img_points = img_points[img_mask]

    camera_center = torch.tensor(
        frame_pcl.kcam.pixel_center, device=img_points.device)

    confidences = torch.norm(
        img_points - camera_center, p=2, dim=1)
    confidences = confidences / confidences.max()

    confidences = torch.exp(-torch.pow(confidences, 2) /
                            (2.0*math.pow(0.6, 2)))

    return confidences


class SurfelCloud:
    """Compact surfels representation in PyTorch.
    """

    def __init__(self, points, colors, normals, radii, confs, times,
                 features, device):
        self.points = points.to(device)
        self.colors = colors.to(device)
        self.normals = normals.to(device)
        self.radii = radii.to(device)
        self.confs = confs.to(device)
        self.times = times.to(device)
        self.features = (features.to(device)
                         if features is not None else None)

    @classmethod
    def from_frame_pcl(cls, frame_pcl, time, device, confs=None, features=None):
        cam_pcl = frame_pcl.unordered_point_cloud(world_space=False)

        if confs is None:
            confs = compute_confidences(frame_pcl)

        radii = compute_surfel_radii(cam_pcl.points, cam_pcl.normals,
                                     frame_pcl.kcam)
        times = torch.full((cam_pcl.points.size(0),), time,
                           dtype=torch.int32).to(device)

        if features is not None:
            features = features[frame_pcl.mask.flatten()].to(device)

        return cls(cam_pcl.points.to(device),
                   cam_pcl.colors.to(device),
                   cam_pcl.normals.to(device),
                   radii.to(device), confs.to(device),
                   times, features, device)

    @property
    def device(self):
        return self.points.device

    def to(self, device):
        return SurfelCloud(self.points, self.colors, self.normals, self.radii,
                           self.confs, self.times, self.features, device)

    def index_select(self, index):
        return SurfelCloud(self.points[index], self.colors[index],
                           self.normals[index], self.radii[index],
                           self.confs[index], self.times[index],
                           (self.features[index]
                            if self.features is not None else None),
                           self.device)

    def transform(self, matrix):
        if self.points.size(0) == 0:
            return

        self.points = Homogeneous(
            matrix.to(self.device)) @ self.points
        normal_matrix = normal_transform_matrix(matrix).to(self.device)
        self.normals = (
            normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    @property
    def size(self):
        return self.points.size(0)


class SurfelModel:
    """Global surfel model on GL buffers.
    """

    def __init__(self, context, max_surfels, device="cuda:0", feature_size=None):
        self.context = context
        self.max_surfels = max_surfels
        self.device = device

        self.active_mask = torch.ones(
            (max_surfels, ), dtype=torch.uint8).to(device)

        self.features = None
        if feature_size is not None:
            self.features = torch.empty((max_surfels, feature_size)).to(device)

        with self.context.current():
            self.points = tenviz.buffer_empty(
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
            with self.points.as_tensor() as points:
                points[new_indices] = new_surfels.points
            with self.colors.as_tensor() as colors:
                colors[new_indices] = new_surfels.colors
            with self.normals.as_tensor() as normals:
                normals[new_indices] = new_surfels.normals
            with self.radii.as_tensor() as radii:
                radii[new_indices] = new_surfels.radii.view(-1, 1)
            with self.confs.as_tensor() as confs:
                confs[new_indices] = new_surfels.confs.view(-1, 1)
            with self.times.as_tensor() as times:
                times[new_indices] = new_surfels.times.view(-1, 1)

        if new_surfels.features is not None:
            self.features[new_indices] = new_surfels.features

    def to_surfel_cloud(self):
        active_idxs = self.get_active_indices()
        with self.context.current():
            points = self.points[active_idxs]
            normals = self.normals[active_idxs]
            colors = self.colors[active_idxs]
            radii = self.radii[active_idxs]
            confs = self.confs[active_idxs]
            times = self.times[active_idxs]

        features = None
        if self.features is not None:
            features = self.features[active_idxs]
        return SurfelCloud(points, colors, normals, radii, confs, times,
                           features, self.device)

    def compact(self):
        """Returns a SurfelModel without free surferls space.
        """
        surfel_cloud = self.to_surfel_cloud()

        compact = SurfelModel(self.context, surfel_cloud.size, self.device,
                              (surfel_cloud.features.size(1)
                               if surfel_cloud.features is not None else None))
        compact.add_surfels(surfel_cloud)
        compact.update_active_mask_gl()
        return compact

    def clone(self):
        clone = SurfelModel(self.context, self.max_surfels, self.device)

        with self.context.current():
            clone.points.from_tensor(self.points.to_tensor())
            clone.normals.from_tensor(self.normals.to_tensor())
            clone.colors.from_tensor(self.colors.to_tensor())
            clone.radii.from_tensor(self.radii.to_tensor())
            clone.confs.from_tensor(self.confs.to_tensor())
            clone.times.from_tensor(self.times.to_tensor())

        if self.features is not None:
            clone.features = self.features.clone()
        else:
            clone.features = None

        clone.active_mask = self.active_mask.clone()
        clone.update_active_mask_gl()

        return clone

    def __str__(self):
        return "SurfelModel with {} active points, {} max. capacity".format(
            self.num_active_surfels(), self.max_surfels)

    def __repr__(self):
        return str(self)
