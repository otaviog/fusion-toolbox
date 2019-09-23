import tenviz

from fiontb.fusion.surfel.model import compute_confidences, compute_surfel_radii
from fiontb.camera import Homogeneous, normal_transform_matrix
from fiontb._cfiontb import MappedSurfelModel

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


class SurfelModel:
    def __init__(self, context, max_surfels, device):
        self.context = context
        self.max_surfels = max_surfels
        self.device = device

        self.active_mask = torch.ones(
            (max_surfels, ), dtype=torch.uint8).to(device)

        with self.context.current():
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
            self.active_mask_gl = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Uint8, integer_attrib=True)
            self.active_mask_gl.from_tensor(self.active_mask)

        self.times = None

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

    def mark_active(self, indices):
        self.active_mask[indices] = 0

    def get_inactive_indices(self, num):
        return self.active_mask.nonzero()[:num].flatten()

    def add_surfels(self, new_surfels, update_gl=False):
        if new_surfels.size == 0:
            return

        new_indices = self.get_inactive_indices(
            new_surfels.size)
        self.mark_active(new_indices)

        with self.context.current():
            with self.map_as_tensors() as mapped:
                mapped.positions[new_indices] = new_surfels.positions
                mapped.colors[new_indices] = new_surfels.colors
                mapped.normals[new_indices] = new_surfels.normals
                mapped.radii[new_indices] = new_surfels.radii
                mapped.confidences[new_indices] = new_surfels.confidences

        if update_gl:
            self.update_gl()

    def update_gl(self):
        with self.context.current():
            self.active_mask_gl.from_tensor(self.active_mask.contiguous())

    def clone(self):
        clone = SurfelModel(self.context, self.max_surfels, self.device)

        with self.context.current():
            clone.positions.from_tensor(self.positions.to_tensor())
            clone.normals.from_tensor(self.normals.to_tensor())
            clone.colors.from_tensor(self.colors.to_tensor())
            clone.radii.from_tensor(self.radii.to_tensor())
            clone.confidences.from_tensor(self.confidences.to_tensor())
            if self.times is not None:
                clone.times.from_tensor(self.times.to_tensor())

        clone.active_mask = self.active_mask.clone()
        clone.update_gl()

        return clone


class LiveSurfels:
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

    @property
    def device(self):
        return self.positions.device

    @property
    def size(self):
        return self.positions.size(0)

    def transform(self, matrix):
        self.positions = Homogeneous(
            matrix.to(self.device)) @ self.positions
        normal_matrix = normal_transform_matrix(matrix).to(self.device)
        self.normals = (
            normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    def to(self, device):
        return LiveSurfels(self.positions.to(device),
                           self.confidences.to(device),
                           self.normals.to(device),
                           self.radii.to(device),
                           self.colors.to(device))

    def __getitem__(self, *args):
        return LiveSurfels(
            self.positions[args],
            self.confidences[args],
            self.normals[args],
            self.radii[args],
            self.colors[args])
