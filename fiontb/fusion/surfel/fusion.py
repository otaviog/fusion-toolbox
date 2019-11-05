import math

import torch

from fiontb.surfel import SurfelCloud
from fiontb.frame import FramePointCloud

from .indexmap import ModelIndexMapRaster, SurfelIndexMapRaster
from .merge_live import MergeLiveSurfels
from .merge import Merge
from .carve_space import CarveSpace
from .clean import Clean
from .stats import FusionStats


class SurfelFusion:
    def __init__(self, model, max_merge_distance=0.005, normal_max_angle=math.radians(30),
                 stable_conf_thresh=10, max_unstable_time=20, search_size=2,
                 indexmap_scale=4, min_z_difference=0.5):
        gl_context = model.gl_context
        self.model = model
        self.model_raster = ModelIndexMapRaster(model)

        self._merge_live_surfels = MergeLiveSurfels(
            gl_context, max_normal_angle=normal_max_angle,
            search_size=search_size)

        self._clean = Clean(
            stable_conf_thresh, max_unstable_time)

        self.indexmap_scale = indexmap_scale
        self._time = 0

    def fuse(self, frame_pcl, rt_cam, features=None, confidence_weight=1.0):
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl, time=self._time,
            features=features, confidence_weight=confidence_weight)

        gl_proj_matrix = frame_pcl.kcam.get_opengl_projection_matrix(
            0.01, 100.0, dtype=torch.float)
        height, width = frame_pcl.image_points.shape[:2]

        if self._time == 0:
            live_surfels.itransform(rt_cam.cam_to_world)
            self.model.add_surfels(live_surfels, update_gl=True)
            self._time += 1
            self.model.max_time = 1
            self.model.max_confidence = live_surfels.confidences.max()

            self._update_pose_indexmap(
                frame_pcl.kcam, rt_cam, gl_proj_matrix, width, height)

            return FusionStats(live_surfels.size, 0, 0)

        stats = FusionStats()

        indexmap_size = int(
            width*self.indexmap_scale), int(height*self.indexmap_scale)
        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_raster.to_indexmap()

        new_surfels = self._merge_live_surfels(
            model_indexmap, live_surfels, gl_proj_matrix,
            rt_cam, width, height, self._time, self.model, live_features=features)
        self.model.add_surfels(new_surfels, update_gl=True)
        stats.added_count = new_surfels.size

        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 indexmap_size[0], indexmap_size[1])
        model_indexmap = self.model_raster.to_indexmap()

        stats.removed_count += self._clean(
            frame_pcl.kcam, frame_pcl.rt_cam,
            model_indexmap, self._time, self.model, update_gl=True)

        self._time += 1
        self.model.max_time = self._time

        return stats

    @property
    def stable_conf_thresh(self):
        return self._clean.stable_conf_thresh
