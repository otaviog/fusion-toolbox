import math

import torch

from fiontb.fusion.surfel.confidence import ConfidenceCache
from fiontb.fusion.surfel.merge_live import MergeLiveSurfels
from fiontb.fusion.surfel.merge import Merge
from fiontb.fusion.surfel.indexmap import ModelIndexMapRaster
from fiontb.fusion.surfel.fusion import FusionStats
from fiontb.surfel import SurfelModel, SurfelCloud
from fiontb.camera import RTCamera

from .registration import SurfelCloudRegistration


class FSFLocalFusion:
    def __init__(self, gl_context, max_surfels, model_indexmap_scale=1.0,
                 feature_size=16, search_size=4, max_normal_angle=math.radians(30),
                 max_merge_distance=0.05):
        self.model = SurfelModel(gl_context, max_surfels,
                                 feature_size=feature_size)
        self.model_indexmap_scale = model_indexmap_scale
        self.model_raster = ModelIndexMapRaster(self.model)
        self.time = 0

        self._conf_cache = ConfidenceCache()
        self._merge_live_surfels = MergeLiveSurfels(gl_context, search_size=search_size,
                                                    max_normal_angle=max_normal_angle)

        self._merge_intern_surfels = Merge(
            max_distance=max_merge_distance,
            normal_max_angle=max_normal_angle,
            search_size=search_size, stable_conf_thresh=0)

        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float))

    def fuse(self, frame_pcl, relative_cam, features, time):
        self.rt_camera = self.rt_camera.integrate(relative_cam.cpu())

        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl,
            confidences=self._conf_cache.get_confidences(frame_pcl),
            features=features, time=time)

        if self.time == 0:
            live_surfels.itransform(relative_cam)
            self.model.add_surfels(live_surfels, update_gl=True)
            self.time = 1
            return FusionStats(live_surfels.size, 0, 0)

        stats = FusionStats()

        gl_proj_matrix = frame_pcl.kcam.get_opengl_projection_matrix(
            0.01, 100.0, dtype=torch.float)
        height, width = frame_pcl.image_points.shape[:2]
        indexmap_size = int(
            width*self.model_indexmap_scale), int(height*self.model_indexmap_scale)

        self.model_raster.raster(gl_proj_matrix, self.rt_camera,
                                 indexmap_size[0], indexmap_size[1])

        with self.model_raster.gl_context.current():
            model_indexmap = self.model_raster.to_indexmap()

        new_surfels = self._merge_live_surfels(
            model_indexmap, live_surfels, gl_proj_matrix,
            self.rt_camera, width, height, self.model)
        self.model.add_surfels(new_surfels, update_gl=True)

        self.model_raster.raster(gl_proj_matrix, self.rt_camera,
                                 indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_raster.to_indexmap()

        stats.merged_count = self._merge_intern_surfels(
            model_indexmap, self.model, update_gl=True)

        return stats

    def reset(self):
        self.time = 0
        self.model.allocator.clear_all()
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float))

    def to_surfel_cloud(self):
        return self.model.to_surfel_cloud()


class FSFFusion:
    def __init__(self, gl_context, max_local_surfels, max_local_frames, max_surfels,
                 feature_size, stable_conf_thresh=10, indexmap_scale=4,
                 registration_iters=45,
                 registration_lr=0.05):

        self.stable_conf_thresh = stable_conf_thresh
        self.time = 0
        self.max_local_frames = max_local_frames

        self.local_fusion = FSFLocalFusion(
            gl_context, max_local_surfels, indexmap_scale, feature_size)

        self.registration = SurfelCloudRegistration(registration_iters,
                                                    registration_lr)

        self.global_model = SurfelModel(gl_context, max_surfels,
                                        feature_size=feature_size)
        self.accum_transform = RTCamera(torch.eye(4, dtype=torch.float))

    def fuse(self, frame_pcl, relative_cam, features=None):
        self.local_fusion.fuse(frame_pcl, relative_cam, features, self.time)
        self.time += 1

        if self.time % self.max_local_frames == 0:
            local_surfels = self.local_fusion.to_surfel_cloud()
            good_conf = local_surfels.confidences > 2
            local_surfels = local_surfels[good_conf]

            self.accum_transform = self.accum_transform.integrate(
                self.local_fusion.rt_camera.matrix)
            if self.global_model.allocated_size == 0:
                self.global_model.add_surfels(local_surfels, update_gl=True)
                self.local_fusion.reset()
                return

            local_surfels.itransform(self.accum_transform.matrix)

            global_surfels = self.global_model.to_surfel_cloud()

            transform = self.registration.estimate(
                global_surfels, local_surfels)
            # self.registration.last_knn_index

            # new_surfels = self._merge_surfels(
            #    local_surfels, self.registration.last_knn_index,
            #    self.global_surfels, global_map)

            local_surfels.itransform(transform)
            self.global_model.add_surfels(local_surfels, update_gl=True)
            # TODO: perform merging
            self.local_fusion.reset()
        return FusionStats()
