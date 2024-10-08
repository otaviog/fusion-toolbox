import math

import torch

from slamtb.surfel import SurfelCloud

from .indexmap import ModelIndexMapRaster
from .update import Update
from .merge import Merge
from .carve_space import CarveSpace
from .clean import Clean
from .stats import FusionStats


class SurfelFusion:
    def __init__(self, model, normal_max_angle=math.radians(30),
                 stable_conf_thresh=10, stable_time_thresh=20,
                 search_size=2, indexmap_scale=4,
                 max_merge_distance=0.01,
                 carve_z_toll=5e-2):
        self.model = model
        self.model_raster = ModelIndexMapRaster(model)

        self._update = Update(
            max_normal_angle=normal_max_angle,
            search_size=search_size,
            elastic_fusion=False)

        self._carve = CarveSpace(stable_conf_thresh=stable_conf_thresh,
                                 stable_time_thresh=stable_time_thresh,
                                 z_tollerance=carve_z_toll)
        self._merge = Merge(max_merge_distance, normal_max_angle=normal_max_angle,
                            search_size=search_size,
                            stable_conf_thresh=stable_conf_thresh)
        self._clean = Clean(elastic_fusion=False,
                            stable_conf_thresh=stable_conf_thresh,
                            stable_time_thresh=stable_time_thresh)

        self.indexmap_scale = indexmap_scale
        self._time = 0

    def fuse(self, frame_pcl, rt_cam, features=None, confidence_weight=1.0,
             sparse_features=None):
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl, time=self._time,
            features=features, confidence_weight=confidence_weight,
            sparse_features=sparse_features)

        gl_proj_matrix = frame_pcl.kcam.get_opengl_projection_matrix(
            0.01, 100.0, dtype=torch.float)
        height, width = frame_pcl.image_points.shape[:2]

        if self._time == 0:
            live_surfels.itransform(rt_cam.cam_to_world)
            self.model.add_surfels(live_surfels, update_gl=True)
            self._time += 1
            self.model.max_time = 1
            self.model.max_confidence = live_surfels.confidences.max()

            return FusionStats(live_surfels.size, 0, 0)

        stats = FusionStats()

        ###
        # Update
        indexmap_size = int(
            width*self.indexmap_scale), int(height*self.indexmap_scale)
        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_raster.to_indexmap()
        new_surfels = self._update(
            model_indexmap, live_surfels, frame_pcl.kcam,
            rt_cam, self._time, self.model)
        self.model.add_surfels(new_surfels, update_gl=True)
        stats.added_count = new_surfels.size

        ####
        # Clean
        stats.removed_count = self._clean(
            frame_pcl.kcam, frame_pcl.rt_cam,
            model_indexmap, self._time, self.model, update_gl=True)

        ####
        # Merge & Carve Raster
        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_raster.to_indexmap()

        # Merge
        stats.merged_count = self._merge(
            model_indexmap, self.model, update_gl=True)

        # Carve
        stats.carved_count += self._carve(frame_pcl.kcam, frame_pcl.rt_cam, model_indexmap,
                                          self._time, self.model)

        self.model.update_gl()
        self._time += 1
        self.model.max_time = self._time

        return stats

    @property
    def stable_conf_thresh(self):
        return self._clean.stable_conf_thresh
