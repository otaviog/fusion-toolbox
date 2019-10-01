import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.fusion.surfel.model import compute_confidences, compute_surfel_radii
from fiontb.camera import RigidTransform, normal_transform_matrix
from fiontb._cfiontb import (
    MappedSurfelModel, SurfelAllocator as _SurfelAllocator)


class SurfelCloud:
    def __init__(self, positions, confidences, normals, radii, colors):
        self.positions = positions
        self.confidences = confidences
        self.normals = normals
        self.radii = radii
        self.colors = colors

    @classmethod
    def from_frame_pcl(cls, frame_pcl, confidences=None):
        pcl = frame_pcl.unordered_point_cloud(world_space=False)
        if confidences is None:
            confidences = compute_confidences(frame_pcl)

        radii = compute_surfel_radii(pcl.points, pcl.normals,
                                     frame_pcl.kcam)

        return cls(pcl.points,
                   confidences,
                   pcl.normals,
                   radii,
                   pcl.colors)

    @classmethod
    def from_frame(cls, frame, confidences=None):
        return cls.from_frame_pcl(FramePointCloud.from_frame(frame), confidences)

    @property
    def device(self):
        return self.positions.device

    @property
    def size(self):
        return self.positions.size(0)

    def itransform(self, matrix):
        self.positions = RigidTransform(
            matrix.to(self.device)) @ self.positions
        normal_matrix = normal_transform_matrix(matrix).to(self.device)
        self.normals = (
            normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    def to(self, device):
        return SurfelCloud(self.positions.to(device),
                           self.confidences.to(device),
                           self.normals.to(device),
                           self.radii.to(device),
                           self.colors.to(device))

    def __getitem__(self, *args):
        return SurfelCloud(
            self.positions[args],
            self.confidences[args],
            self.normals[args],
            self.radii[args],
            self.colors[args])


class _MappedSurfelModelContext:
    def __init__(self, positions_map,
                 confidences_map,
                 normals_map,
                 radii_map,
                 colors_map):
        self.positions_map = positions_map
        self.confidences_map = confidences_map
        self.normals_map = normals_map
        self.radii_map = radii_map
        self.colors_map = colors_map

    def __enter__(self):
        params = MappedSurfelModel()
        params.positions = self.positions_map.tensor
        params.confidences = self.confidences_map.tensor.squeeze()
        params.normals = self.normals_map.tensor
        params.radii = self.radii_map.tensor.squeeze()
        params.colors = self.colors_map.tensor

        return params

    def __exit__(self, *args):
        self.positions_map.unmap()
        self.confidences_map.unmap()
        self.normals_map.unmap()
        self.radii_map.unmap()
        self.colors_map.unmap()


class _CPUCopySurfelModelContext:
    def __init__(self, positions_buff,
                 confidences_buff,
                 normals_buff,
                 radii_buff,
                 colors_buff):
        self.positions_buff = positions_buff
        self.confidences_buff = confidences_buff
        self.normals_buff = normals_buff
        self.radii_buff = radii_buff
        self.colors_buff = colors_buff
        self._params = None

    def __enter__(self):
        params = MappedSurfelModel()
        params.positions = self.positions_buff.to_tensor(False)
        params.confidences = self.confidences_buff.to_tensor(False).squeeze()
        params.normals = self.normals_buff.to_tensor(False)
        params.radii = self.radii_buff.to_tensor(False).squeeze()
        params.colors = self.colors_buff.to_tensor(False)

        self._params = params
        return params

    def __exit__(self, *args):
        params = self._params
        self.positions_buff.from_tensor(params.positions)
        self.confidences_buff.from_tensor(params.confidences)
        self.normals_buff.from_tensor(params.normals)
        self.radii_buff.from_tensor(params.radii)
        self.colors_buff.from_tensor(params.colors)


class SurfelAllocator(_SurfelAllocator):
    def __init__(self, max_surfels):
        super(SurfelAllocator, self).__init__(max_surfels, "cuda:0")
        self.max_surfels = max_surfels
        self.active_count = 0

    def mark_active(self, indices):
        self.free_mask_byte[indices] = 0
        self.free_mask_bit[indices] = False
        self.active_count += indices.size(0)

    def mark_unactive(self, indices):
        self.free_mask_byte[indices] = 1
        self.free_mask_bit[indices] = True
        self.active_count -= indices.size(0)

    def allocate(self, num_elements):
        unactive = torch.empty(num_elements, dtype=torch.int64)
        self.find_unactive(unactive)
        self.mark_active(unactive)
        return unactive

    def clear_all(self):
        self.free_mask_byte[:] = 1
        self.free_mask_bit[:] = True
        self.active_count = 0

    def copy_(self, other):
        self.free_mask_byte = other.free_mask_byte.clone()
        self.free_mask_bit = other.free_mask_bit.clone()
        self.active_count = other.active_count


class SurfelModel:
    def __init__(self, gl_context, max_surfels):
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
            self.free_mask_gl = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Uint8, integer_attrib=True)
            self.free_mask_gl.from_tensor(self.allocator.free_mask_byte)

        self.times = None

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
                self.colors.as_tensor_())

        return _CPUCopySurfelModelContext(
            self.positions,
            self.confidences,
            self.normals,
            self.radii,
            self.colors)

    def to_surfel_cloud(self):
        active_mask = self.allocator.free_mask_byte == 0
        with self.gl_context.current():
            with self.map_as_tensors() as mapped:
                return SurfelCloud(mapped.positions[active_mask].clone(),
                                   mapped.confidences[active_mask].clone(),
                                   mapped.normals[active_mask].clone(),
                                   mapped.radii[active_mask].clone(),
                                   mapped.colors[active_mask].clone())

    def mark_unactive(self, indices):
        self.allocator.mark_unactive(indices)

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

            if update_gl:
                self.free_mask_gl.from_tensor(self.allocator.free_mask_byte)

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
            if self.times is not None:
                clone.times.from_tensor(self.times.to_tensor())

        clone.allocator.copy_(self.allocator)
        clone.update_gl()

        return clone

    @property
    def device(self):
        # TODO: unhard-code device
        return "cuda:0"

    @property
    def max_surfels(self):
        return self.allocator.max_surfels
