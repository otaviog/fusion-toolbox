import torch

from fiontb.fusion.surfel.confidence import ConfidenceCache
from fiontb.fusion.surfel.merge_live import MergeLiveSurfels
from fiontb.fusion.surfel.indexmap import ModelIndexMapRaster
from fiontb.fusion.surfel.fusion import FusionStats
from fiontb.surfel import SurfelModel, SurfelCloud

from .registration import SurfelCloudRegistration


class FSFLocalFusion:
    def __init__(self, gl_context, max_surfels, model_indexmap_scale=1.0):
        self.model = SurfelModel(gl_context, max_surfels)
        self.model_indexmap_scale = model_indexmap_scale
        self.model_raster = ModelIndexMapRaster(self.model)
        self.time = 0

        self._conf_cache = ConfidenceCache()
        self._merge_live_surfels = MergeLiveSurfels(gl_context)

    def fuse(self, frame_pcl, rt_cam):
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl,
            confidences=self._conf_cache.get_confidences(frame_pcl))

        if self.time == 0:
            live_surfels.itransform(rt_cam.cam_to_world)
            self.model.add_surfels(live_surfels, update_gl=True)
            self.time = 1
            return

        gl_proj_matrix = frame_pcl.kcam.get_opengl_projection_matrix(
            0.01, 100.0, dtype=torch.float)
        height, width = frame_pcl.image_points.shape[:2]

        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 int(width*self.model_indexmap_scale),
                                 int(height*self.model_indexmap_scale))

        with self.model_raster.gl_context.current():
            model_indexmap = self.model_raster.to_indexmap()

        new_surfels = self._merge_live_surfels(
            model_indexmap, live_surfels, gl_proj_matrix,
            rt_cam, width, height, self.model)
        self.model.add_surfels(new_surfels, update_gl=True)

    def reset(self):
        self.time = 0
        self.model.allocator.clear_all()

    def to_surfel_cloud(self):
        return self.model.to_surfel_cloud()


class FSFFusion:
    def __init__(self, gl_context, max_local_surfels, max_local_frames, max_surfels,
                 stable_conf_thresh=10, indexmap_scale=4,
                 registration_iters=45,
                 registration_lr=0.05):

        self.stable_conf_thresh = stable_conf_thresh
        self.time = 0
        self.max_local_frames = max_local_frames

        self.local_fusion = FSFLocalFusion(
            gl_context, max_local_surfels, indexmap_scale)

        self.registration = SurfelCloudRegistration(registration_iters,
                                                    registration_lr)

        self.global_model = SurfelModel(gl_context, max_surfels)

    def fuse(self, frame_pcl, rt_cam):
        self.time += 1

        if self.time % self.max_local_frames != 0:
            self.local_fusion.fuse(frame_pcl, rt_cam)
        else:
            local_surfels = self.local_fusion.to_surfel_cloud()
            if self.global_model.allocator.active_count == 0:
                self.global_model.add_surfels(local_surfels, update_gl=True)
                return

            global_surfels = self.global_model.to_surfel_cloud()
            transform = self.registration.estimate(
                global_surfels, local_surfels)
            local_surfels.itransform(transform)

            self.global_model.add_surfels(local_surfels, update_gl=True)
            # TODO: perform merging
            self.local_fusion.reset()
        return FusionStats()
