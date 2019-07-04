import torch
import tenviz

from .model import compute_confidences, SurfelCloud
from .live_merge import LiveToModelMergeMap
from .spacecarving import SpaceCarving
from .intra_merge import IntraModelMergeMap
from .indexmap import ModelIndexMap


class _ConfidenceCache:
    def __init__(self):
        self.width = -1
        self.height = -1
        self.confidences = None

    def get_confidences(self, frame_pcl):
        fheight, fwidth = frame_pcl.image_points.shape[:2]
        # It doesn't check kcam
        if fheight != self.height or fwidth != self.width:
            self.width = fwidth
            self.height = fheight
            self.confidences = compute_confidences(frame_pcl, no_mask=True)

        return self.confidences[frame_pcl.fg_mask.flatten()]


class FusionStats:
    """Surfel fusion step statistics.

    Attributes:

       added_count (int): How many surfels were added in the step.

       merged_count (int): How many surfels were merged in the step.

       removed_count (int): How many surfels were removed in the step.
    """

    def __init__(self, added_count, merged_count, removed_count):
        self.added_count = added_count
        self.merged_count = merged_count
        self.removed_count = removed_count

    def __str__(self):
        return "Fusion stats: {} added, {} merged, {} removed".format(
            self.added_count, self.merged_count, self.removed_count)

    def __repr__(self):
        return str(self)


class SurfelFusion:
    def __init__(self, surfels, max_distance=0.05, normal_max_angle=20.0,
                 stable_conf_thresh=10, max_unstable_time=20):
        self.surfels = surfels
        self.live_merge_map = LiveToModelMergeMap(surfels)
        self.intra_merge_map = IntraModelMergeMap(surfels)
        self.pose_indexmap = ModelIndexMap(surfels)
        self.spacecarving = SpaceCarving(surfels)

        self.max_distance = max_distance
        self.normal_min_dot = 1 - normal_max_angle / 90.0

        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time

        self._is_first_fusion = True

        self._time = 0
        self.merge_min_radio = 0.5

        self._conf_compute_cache = _ConfidenceCache()

    def fuse(self, frame_pcl, rt_cam, features=None):
        device = "cuda:0"

        frame_confs = self._conf_compute_cache.get_confidences(frame_pcl)
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl, self._time, device, confs=frame_confs, features=features)

        proj_matrix = torch.from_numpy(tenviz.projection_from_kcam(
            frame_pcl.kcam.matrix, 0.01, 10.0).to_matrix()).float()
        height, width = frame_pcl.image_points.shape[:2]

        if self._is_first_fusion:
            live_surfels.transform(rt_cam.cam_to_world)
            self.surfels.add_surfels(live_surfels)
            self.surfels.update_active_mask_gl()
            self._is_first_fusion = False

            self._update_cam_view(proj_matrix, rt_cam, width, height)
            return FusionStats(live_surfels.size, 0, 0)

        live_query = self.live_merge_map.find_mergeable(
            live_surfels, proj_matrix, rt_cam, width, height)
        live_idxs, model_idxs, live_unst_idxs, visible_model_idxs = live_query

        live_surfels.transform(rt_cam.cam_to_world)
        if live_unst_idxs.size(0) > 0:
            self.surfels.add_surfels(live_surfels.index_select(live_unst_idxs))

        with self.surfels.context.current():
            model_surfels = SurfelCloud(
                self.surfels.points[model_idxs],
                self.surfels.colors[model_idxs],
                self.surfels.normals[model_idxs],
                self.surfels.radii[model_idxs].squeeze(),
                self.surfels.confs[model_idxs].squeeze(),
                self.surfels.times[model_idxs].squeeze(),
                (self.surfels.features[model_idxs]
                 if self.surfels.features is not None else None),
                device)

        live_idxs = live_idxs.to(device)
        self._merge_surfels(
            live_surfels.index_select(live_idxs),
            model_surfels, model_idxs.to(device))

        removed_count = self._remove_surfels(visible_model_idxs)
        self.surfels.update_active_mask_gl()

        active_count = self.surfels.num_active_surfels()

        fb_scale = 4
        self.spacecarving.carve(proj_matrix, rt_cam, int(width*fb_scale), int(height*fb_scale),
                                self.stable_conf_thresh, self._time, 4)

        self.surfels.update_active_mask_gl()

        self.intra_merge_map.merge_close_surfels(proj_matrix, rt_cam,
                                                 int(width*fb_scale), int(height*fb_scale),
                                                 self.stable_conf_thresh)

        removed_count += active_count - self.surfels.num_active_surfels()
        self._time += 1
        self.surfels.max_time = self._time

        self.surfels.update_active_mask_gl()
        self._update_cam_view(proj_matrix, rt_cam, width, height)
        return FusionStats(live_unst_idxs.size(0), model_idxs.size(0),
                           removed_count)

    def _merge_surfels(self, live, model, model_idxs):
        # radii_mask = (live.radii < model.radii *
        #              (1.0 + self.merge_min_radio))
        # live_idxs = live_idxs[radii_mask]
        # model_idxs = model_idxs[radii_mask]

        # confs = model.confs[radii_mask].view(-1, 1)
        # confs_update = confs_update[radii_mask].view(-1, 1)
        # live_radii = live.radii[radii_mask].view(-1, 1)

        confs_update = (model.confs + live.confs).view(-1, 1)
        model.points = (model.points * model.confs.view(-1, 1) + live.points *
                        live.confs.view(-1, 1)) / confs_update
        model.colors = (model.colors.float() * model.confs.view(-1, 1) + live.colors.float() *
                        live.confs.view(-1, 1)) / confs_update
        model.colors = model.colors.byte()

        model.normals = (model.normals * model.confs.view(-1, 1) +
                         live.normals*live.confs.view(-1, 1)) / confs_update
        model.normals /= model.normals.norm(2, 1).view(-1, 1)
        model.confs = confs_update.squeeze()

        with self.surfels.context.current():
            self.surfels.points[model_idxs] = model.points
            self.surfels.colors[model_idxs] = model.colors
            self.surfels.normals[model_idxs] = model.normals
            self.surfels.confs[model_idxs] = model.confs
            self.surfels.times[model_idxs] = live.times

    def _remove_surfels(self, visible_model_idxs):
        with self.surfels.context.current():
            confs = self.surfels.confs[visible_model_idxs].squeeze()
            times = self.surfels.times[visible_model_idxs].squeeze()

        unstable_idxs = visible_model_idxs[(confs < self.stable_conf_thresh)
                                           & (self._time - times >= self.max_unstable_time)]

        self.surfels.mark_inactive(unstable_idxs)

        return unstable_idxs.size(0)

    def get_stable_points(self):
        active_idxs = self.surfels.get_active_indices()
        with self.surfels.context.current():
            confs = self.surfels.confs[active_idxs].squeeze()
            stable_idxs = active_idxs[confs < self.stable_conf_thresh]

            points = self.surfels.points[stable_idxs]

        return points

    def _update_cam_view(self, proj_matrix, rt_cam, width, height):
        self.pose_indexmap.raster(proj_matrix, rt_cam, width, height, -1.0, -1)

