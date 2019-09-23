import torch

from fiontb.fusion.surfel.fusion import _ConfidenceCache
from fiontb._cfiontb import FeatSurfel
from .datatype import SurfelModel, LiveSurfels


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

        proj_matrix = frame_pcl.kcam.get_opengl_projection_matrix(
            0.01, 10.0, dtype=torch.float)
        height, width = frame_pcl.image_points.shape[:2]

        if self.time == 0:
            live_surfels.transform(rt_cam.cam_to_world)
            self.model.add_surfels(live_surfels)
            self.model.update_gl()

            self.time = 1


class FeatSurfelGlobalFusion:
    def fuse(self, frame_pcl, rt_cam):
        if self._count % self.K != 0:
            self.local_fusion.fuse(frame_pcl, rt_cam)
        else:
            local_model = self.local_fusion.to_surfel_model()
            # align local_model with global_model
