import torch

from slamtb.surfel import SurfelCloud

from .update import Update
from .clean import Clean
from .stats import FusionStats
from .indexmap import ModelIndexMapRaster


class EFFusion:
    def __init__(self, model, stable_conf_thresh=10, stable_time_thresh=20,
                 search_size=2, indexmap_scale=4):
        self.model = model
        self.model_raster = ModelIndexMapRaster(model)

        self._update = Update(elastic_fusion=True, search_size=search_size)
        self._clean = Clean(elastic_fusion=True,
                            stable_conf_thresh=stable_conf_thresh,
                            stable_time_thresh=stable_time_thresh)

        self.indexmap_scale = indexmap_scale
        self._time = 0

    def fuse(self, frame_pcl, rt_cam, features=None, confidence_weight=1.0):
        device = frame_pcl.device
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

            return FusionStats(live_surfels.size, 0, 0)

        stats = FusionStats(merged_count="n/a", carved_count="n/a")

        indexmap_size = int(
            width*self.indexmap_scale), int(height*self.indexmap_scale)
        self.model_raster.raster(gl_proj_matrix, rt_cam,
                                 indexmap_size[0], indexmap_size[1])

        model_indexmap = self.model_raster.to_indexmap(device=device)
        # import cv2
        # cv2.imshow("indexmap", model_indexmap.color.cpu().numpy())

        new_surfels = self._update(
            model_indexmap, live_surfels, frame_pcl.kcam,
            rt_cam, self._time, self.model)
        self.model.add_surfels(new_surfels, update_gl=True)
        stats.added_count = new_surfels.size

        stats.removed_count += self._clean(
            frame_pcl.kcam, frame_pcl.rt_cam,
            model_indexmap, self._time, self.model, update_gl=True)

        self._time += 1
        self.model.max_time = self._time

        return stats

    @property
    def stable_conf_thresh(self):
        return self._clean.stable_conf_thresh
