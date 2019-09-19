import torch
import tenviz

from fiontb.fusion.surfel.model import compute_confidences, compute_surfel_radii
from fiontb.fusion.surfel.fusion import _ConfidenceCache
from fiontb.camera import Homogeneous, normal_transform_matrix
from fiontb._cfiontb import FeatSurfel, SurfelModelParams


class _SurfelModelParamsContext:
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
        params = SurfelModelParams()
        params.positions = self.positions_map.tensor
        params.confidences = self.confidences_map.tensor
        params.normals = self.normals_map.tensor
        params.radii = self.radii_map.tensor
        params.colors = self.colors_map.tensor

        return params

    def __exit__(self, *args):
        self.positions_map.unmap()
        self.confidences_map.unmap()
        self.normals_map.unmap()
        self.radii_map.unmap()
        self.colors_map.unmap()


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
            self.confs = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Float)
            self.active_mask_gl = tenviz.buffer_empty(
                max_surfels, 1, tenviz.BType.Uint8, integer_attrib=True)
            self.active_mask_gl.from_tensor(self.active_mask)

    def as_params(self):
        return _SurfelModelParamsContext(
            self.positions.as_tensor_(),
            self.confs.as_tensor_(),
            self.normals.as_tensor_(),
            self.radii.as_tensor_(),
            self.colors.as_tensor_())

    def mark_active(self, indices):
        self.active_mask[indices] = 0

    def get_inactive_indices(self, num):
        return self.active_mask.nonzero()[:num].flatten()

    def add_surfels(self, new_surfels):
        new_indices = self.get_inactive_indices(
            new_surfels.size)
        self.mark_active(new_indices)

        with self.context.current():
            with self.positions.as_tensor() as positions:
                positions[new_indices] = new_surfels.positions
            with self.colors.as_tensor() as colors:
                colors[new_indices] = new_surfels.colors
            with self.normals.as_tensor() as normals:
                normals[new_indices] = new_surfels.normals
            with self.radii.as_tensor() as radii:
                radii[new_indices] = new_surfels.radii.view(-1, 1)
            with self.confs.as_tensor() as confs:
                confs[new_indices] = new_surfels.confs.view(-1, 1)

    def update_gl(self):
        with self.context.current():
            self.active_mask_gl.from_tensor(self.active_mask.contiguous())


class LiveSurfels:
    def __init__(self, positions, confidences, normals, radii, colors):
        self.positions = positions
        self.confidences = confidences
        self.normals = normals
        self.radii = radii
        self.colors = colors

    @classmethod
    def from_frame_pcl(cls, frame_pcl, confidences):
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


class FeatSurfelLocalFusion:
    def __init__(self, tv_context, max_surfels, indexmap_scale, device):
        self._conf_cache = _ConfidenceCache()
        self.model = SurfelModel(tv_context, max_surfels, device)
        self.time = 0
        self.indexmap_scale = indexmap_scale

    def fuse(self, frame_pcl, rt_cam):
        live_surfels = LiveSurfels.from_frame_pcl(
            frame_pcl,
            confidences=self._conf_cache.get_confidences(frame_pcl))

        proj_matrix = torch.from_numpy(tenviz.projection_from_kcam(
            frame_pcl.kcam.matrix, 0.01, 10.0).to_matrix()).float()
        height, width = frame_pcl.image_points.shape[:2]

        if self.time == 0:
            live_surfels.transform(rt_cam.cam_to_world)
            self.model.add_surfels(live_surfels)
            self.model.update_gl()

            self.time = 1
            return None

        self.live_indexmap.raster(live_surfels, proj_matrix, width, height)
        live_indexmap = self.live_indexmap.to_params()

        indexmap_size = int(
            width*self.indexmap_scale), int(height*self.indexmap_scale)

        self.model_indexmap.raster(proj_matrix, rt_cam,
                                   indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_indexmap.to_params()

        with self.model.as_params() as model:
            FeatSurfel.merge_live(model_indexmap, live_indexmap, model)


class FeatSurfelGlobalFusion:
    def fuse(self, frame_pcl, rt_cam):
        if self._count % self.K != 0:
            self.local_fusion.fuse(frame_pcl, rt_cam)
        else:
            local_model = self.local_fusion.to_surfel_model()
            # align local_model with global_model
