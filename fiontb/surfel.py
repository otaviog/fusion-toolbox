import math

import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.camera import RigidTransform, normal_transform_matrix
from fiontb._cfiontb import (
    SurfelOp as _SurfelOp,
    MappedSurfelModel, SurfelAllocator as _SurfelAllocator,
    SurfelCloud as CppSurfelCloud)
import fiontb._utils as _utils


def compute_surfel_radii(cam_points, normals, kcam):
    focal_len = (abs(kcam.matrix[0, 0]) + abs(kcam.matrix[1, 1])) * .5
    radii = (
        cam_points[:, 2] / focal_len) * math.sqrt(2)
    radii = torch.min(2*radii, radii /
                      normals[:, 2].squeeze().abs())

    return radii


class ComputeConfidences:
    def __init__(self):
        self._confidences = None

    def __call__(self, kcam, weight, width, height):
        center = kcam.pixel_center
        max_dist = math.sqrt((width - center[0])*(width - center[0])
                             + (height - center[1])*(height - center[1]))

        self._confidences = _utils.empty_ensured_size(self._confidences, height, width,
                                                      dtype=torch.float,
                                                      device=kcam.device)
        _SurfelOp.compute_confidences(
            kcam.matrix, weight, max_dist, self._confidences)
        return self._confidences


def compute_confidences(frame_pcl, no_mask=False):
    img_points = frame_pcl.image_points[:, :, :2].view(-1, 2)
    img_mask = frame_pcl.mask.flatten()

    if not no_mask:
        img_points = img_points[img_mask]

    camera_center = torch.tensor(
        frame_pcl.kcam.pixel_center, device=img_points.device)

    confidences = torch.norm(
        img_points - camera_center, p=2, dim=1)
    confidences = confidences / 400  # confidences.max()

    confidences = torch.exp(-torch.pow(confidences, 2) /
                            (2.0*math.pow(0.6, 2)))

    return confidences


class SurfelCloud:
    def __init__(self, positions, confidences, normals, radii,
                 colors, times, features=None):
        self.positions = positions
        self.confidences = confidences
        self.normals = normals
        self.radii = radii
        self.colors = colors
        self.times = times
        self.features = features

    @classmethod
    def from_frame_pcl(cls, frame_pcl, confidences=None, confidence_weight=0.5,
                       time=0, features=None):
        pcl = frame_pcl.unordered_point_cloud(world_space=False)
        if confidences is None:
            confidences = ComputeConfidences()(
                frame_pcl.kcam, confidence_weight,
                frame_pcl.width, frame_pcl.height)[frame_pcl.mask]

        radii = compute_surfel_radii(pcl.points, pcl.normals,
                                     frame_pcl.kcam)

        if isinstance(time, int):
            time = torch.full((pcl.size, ), time, dtype=torch.int32,
                              device=frame_pcl.points.device)

        if features is not None:
            features = features[:, frame_pcl.mask].view(-1, pcl.size)

        return cls(pcl.points,
                   confidences,
                   pcl.normals,
                   radii,
                   pcl.colors,
                   time, features)

    @classmethod
    def from_frame(cls, frame, confidences=None, confidence_weight=0.5,
                   time=0, features=None):
        return cls.from_frame_pcl(FramePointCloud.from_frame(frame),
                                  confidences=confidences,
                                  confidence_weight=confidence_weight,
                                  time=time,
                                  features=features)

    @property
    def device(self):
        return self.positions.device

    @property
    def size(self):
        return self.positions.size(0)

    @property
    def has_features(self):
        return self.features is not None

    @property
    def feature_size(self):
        return self.features.size(0)

    def itransform(self, matrix):
        if False:
            self.positions = RigidTransform(
                matrix.to(self.device)) @ self.positions
            normal_matrix = normal_transform_matrix(matrix).to(self.device)
            self.normals = (
                normal_matrix @ self.normals.view(-1, 3, 1)).squeeze()
        else:
            transform = RigidTransform(matrix.to(self.device))
            transform.inplace(self.positions)
            transform.inplace_normals(self.normals)

    def transform(self, matrix):
        normal_matrix = normal_transform_matrix(matrix).to(self.device)

        return SurfelCloud(RigidTransform(matrix.to(self.device)) @ self.positions,
                           self.confidences,
                           (normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze(),
                           self.radii, self.colors, self.times)

    def to(self, device):
        return SurfelCloud(self.positions.to(device),
                           self.confidences.to(device),
                           self.normals.to(device),
                           self.radii.to(device),
                           self.colors.to(device),
                           self.times.to(device),
                           features=self.features.to(device)
                           if self.features is not None else None)

    def to_open3d(self):
        from open3d import PointCloud as o3dPointCloud, Vector3dVector

        pcl = o3dPointCloud()

        pcl.points = Vector3dVector(self.positions.squeeze().cpu().numpy())
        pcl.colors = Vector3dVector(self.colors.squeeze().cpu().numpy())
        pcl.normals = Vector3dVector(self.normals.cpu().numpy())

        return pcl

    def to_cpp_(self):
        params = CppSurfelCloud()
        params.positions = self.positions
        params.confidences = self.confidences
        params.normals = self.normals
        params.radii = self.radii
        params.colors = self.colors
        params.times = self.times
        if self.features is not None:
            params.features = self.features
        else:
            params.features = torch.empty(
                0, 0, dtype=torch.float,
                device=self.device)
        return params

    def __getitem__(self, *args):
        return SurfelCloud(
            self.positions[args],
            self.confidences[args],
            self.normals[args],
            self.radii[args],
            self.colors[args],
            self.times[args],
            features=self.features[:, args[0]]
            if self.features is not None else None)


class _MappedSurfelModelContext:
    def __init__(self, positions_map,
                 confidences_map,
                 normals_map,
                 radii_map,
                 colors_map,
                 times_map, features):
        self.positions_map = positions_map
        self.confidences_map = confidences_map
        self.normals_map = normals_map
        self.radii_map = radii_map
        self.colors_map = colors_map
        self.times_map = times_map
        self.features = features

    def __enter__(self):
        params = MappedSurfelModel()
        params.positions = self.positions_map.tensor
        params.confidences = self.confidences_map.tensor.squeeze()
        params.normals = self.normals_map.tensor
        params.radii = self.radii_map.tensor.squeeze()
        params.colors = self.colors_map.tensor
        params.times = self.times_map.tensor.squeeze()

        if self.features is not None:
            params.features = self.features
        else:
            params.features = torch.empty(
                (0, 0),
                device=params.positions.device,
                dtype=torch.float)

        return params

    def __exit__(self, *args):
        self.positions_map.unmap()
        self.confidences_map.unmap()
        self.normals_map.unmap()
        self.radii_map.unmap()
        self.colors_map.unmap()
        self.times_map.unmap()


class _CPUCopySurfelModelContext:
    def __init__(self, positions_buff,
                 confidences_buff,
                 normals_buff,
                 radii_buff,
                 colors_buff,
                 times_buff, features):
        self.positions_buff = positions_buff
        self.confidences_buff = confidences_buff
        self.normals_buff = normals_buff
        self.radii_buff = radii_buff
        self.colors_buff = colors_buff
        self.times_buff = times_buff
        self.features = features
        self._params = None

    def __enter__(self):
        params = MappedSurfelModel()
        params.positions = self.positions_buff.to_tensor(False)
        params.confidences = self.confidences_buff.to_tensor(False).squeeze()
        params.normals = self.normals_buff.to_tensor(False)
        params.radii = self.radii_buff.to_tensor(False).squeeze()
        params.colors = self.colors_buff.to_tensor(False)
        params.times = self.times_buff.to_tensor(False).squeeze()

        if self.features is not None:
            params.features = self.features.cpu()
        else:
            params.features = torch.empty(
                (0, 0), dtype=torch.float)

        self._params = params
        return params

    def __exit__(self, *args):
        params = self._params
        self.positions_buff.from_tensor(params.positions)
        self.confidences_buff.from_tensor(params.confidences)
        self.normals_buff.from_tensor(params.normals)
        self.radii_buff.from_tensor(params.radii)
        self.colors_buff.from_tensor(params.colors)
        self.times_buff.from_tensor(params.times)
        if self.features is not None:
            self.features[:] = params.features


class SurfelAllocator(_SurfelAllocator):
    def __init__(self, max_surfels):
        super().__init__(max_surfels)
        self.free_mask_byte = torch.ones((max_surfels),
                                         dtype=torch.uint8,
                                         device="cuda:0")

    def free(self, indices):
        super().free(indices.cpu())
        self.free_mask_byte[indices] = 1

    def allocate(self, num_elements):
        indices = torch.empty(num_elements, dtype=torch.int64)
        super().allocate(indices)
        self.free_mask_byte[indices] = 0
        return indices

    def allocated_indices(self):
        return (self.free_mask_byte == 0).nonzero().squeeze()

    def clear_all(self):
        self.free_mask_byte[:] = 1
        self.free_all()

    def copy_(self, other):
        self.free_mask_byte = other.free_mask_byte.clone()
        super().copy_(other)


class SurfelModel:
    def __init__(self, gl_context, max_surfels, max_confidence=0, max_time=0, feature_size=0):
        self.gl_context = gl_context
        self.allocator = SurfelAllocator(max_surfels)

        with self.gl_context.current():
            self.positions = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.normals = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Float)
            self.colors = tenviz.buffer_empty(
                max_surfels, 3, tenviz.BType.Uint8, normalize=True)

            self.radii = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.confidences = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)

            self.times = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Int32,
                integer_attrib=True)

            self.free_mask_gl = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Uint8, integer_attrib=True)
            self.free_mask_gl.from_tensor(self.allocator.free_mask_byte)

        if feature_size > 0:
            self.features = torch.empty(
                feature_size, max_surfels, device=self.device)
        else:
            self.features = None

        self.max_confidence = max_confidence
        self.max_time = max_time

    @classmethod
    def from_surfel_cloud(cls, gl_context, surfels):
        model = cls(gl_context, surfels.size)

        model.add_surfels(surfels, update_gl=True)

        return model

    def map_as_tensors(self, device=None):
        if device is None or torch.device(device).type != 'cpu':
            return _MappedSurfelModelContext(
                self.positions.as_tensor_(),
                self.confidences.as_tensor_(),
                self.normals.as_tensor_(),
                self.radii.as_tensor_(),
                self.colors.as_tensor_(),
                self.times.as_tensor_(), self.features)

        return _CPUCopySurfelModelContext(
            self.positions,
            self.confidences,
            self.normals,
            self.radii,
            self.colors,
            self.times, self.features)

    def to_surfel_cloud(self):
        active_mask = self.allocator.free_mask_byte == 0
        active_indices = active_mask.nonzero().to(self.device).squeeze()
        with self.gl_context.current():
            with self.map_as_tensors() as mapped:
                cloud = SurfelCloud(mapped.positions[active_indices].clone(),
                                    mapped.confidences[active_indices].clone(),
                                    mapped.normals[active_indices].clone(),
                                    mapped.radii[active_indices].clone(),
                                    mapped.colors[active_indices].clone(),
                                    mapped.times[active_indices].clone(),
                                    features=self.features[:, active_indices]
                                    if self.features is not None else None)
        return cloud, active_indices

    def free(self, indices, update_gl=False):
        self.allocator.free(indices)
        if update_gl:
            self.update_gl()

    def add_surfels(self, new_surfels, update_gl=False):
        if new_surfels.size == 0:
            return

        new_indices = self.allocator.allocate(new_surfels.size)
        new_surfels = new_surfels.to(self.device)
        with self.gl_context.current():
            with self.map_as_tensors() as mapped:
                mapped.positions[new_indices] = new_surfels.positions
                mapped.colors[new_indices] = new_surfels.colors
                mapped.normals[new_indices] = new_surfels.normals
                mapped.radii[new_indices] = new_surfels.radii
                mapped.confidences[new_indices] = new_surfels.confidences
                mapped.times[new_indices] = new_surfels.times

            if update_gl:
                self.free_mask_gl.from_tensor(self.allocator.free_mask_byte)
        if self.features is not None and new_surfels.features is not None:
            self.features[:, new_indices] = new_surfels.features

    def update_gl(self):
        with self.gl_context.current():
            self.free_mask_gl.from_tensor(self.allocator.free_mask_byte)

    def clone(self):
        clone = SurfelModel(self.gl_context, self.max_surfels)

        with self.gl_context.current():
            clone.positions.from_tensor(self.positions.to_tensor())
            clone.normals.from_tensor(self.normals.to_tensor())
            clone.colors.from_tensor(self.colors.to_tensor())
            clone.radii.from_tensor(self.radii.to_tensor())
            clone.confidences.from_tensor(self.confidences.to_tensor())
            clone.times.from_tensor(self.times.to_tensor())
            if self.features is not None:
                clone.features = self.features.clone()

        clone.allocator.copy_(self.allocator)
        clone.update_gl()

        return clone

    def allocated_indices(self):
        return self.allocator.allocated_indices()

    def clear(self):
        self.allocator.clear_all()

    @property
    def device(self):
        # TODO: unhard-code device
        return "cuda:0"

    @property
    def max_surfels(self):
        return self.allocator.max_size

    @property
    def allocated_size(self):
        return self.allocator.allocated_size

    @property
    def has_features(self):
        return self.features is not None

    @property
    def feature_size(self):
        return self.features.size(0)
