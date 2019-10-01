import math

import torch

import tenviz

import fiontb.frame
from fiontb.surfel import SurfelCloud
from .merge_live import LiveToModelMergeMap
from .spacecarving import SpaceCarving
from .intra_merge import IntraMergeMap
from .indexmap import ModelIndexMap


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
    def __init__(self, surfels, max_distance=0.005, normal_max_angle=math.radians(30),
                 stable_conf_thresh=10, max_unstable_time=20,
                 indexmap_scale=4):

        self.surfels = surfels
        self.model_indexmap = ModelIndexMap(surfels)
        self.live_merge_map = LiveToModelMergeMap(surfels,
                                                  normal_max_angle=normal_max_angle,
                                                  search_size=2)
        self.pose_indexmap = ModelIndexMap(surfels)

        self.intra_merge_map = IntraMergeMap(max_dist=max_distance,
                                             normal_max_angle=normal_max_angle,
                                             search_size=2)
        self.spacecarving = SpaceCarving(surfels, stable_conf_thresh,
                                         search_size=2, min_z_difference=0.1)

        self.stable_conf_thresh = stable_conf_thresh
        self.max_unstable_time = max_unstable_time
        self.indexmap_scale = indexmap_scale

        self._time = 0
        self._conf_compute_cache = _ConfidenceCache()

        self._last_kcam = None
        self._last_rtcam = None

        self._merge_min_radio = 0.5

    def fuse(self, frame_pcl, rt_cam, features=None):
        device = "cuda:0"

        frame_confs = self._conf_compute_cache.get_confidences(frame_pcl)
        live_surfels = SurfelCloud.from_frame_pcl(
            frame_pcl, self._time, device, confs=frame_confs, features=features)

        proj_matrix = torch.from_numpy(tenviz.projection_from_kcam(
            frame_pcl.kcam.matrix, 0.01, 10.0).to_matrix()).float()
        height, width = frame_pcl.image_points.shape[:2]

        if self._time == 0:
            live_surfels.transform(rt_cam.cam_to_world)
            self.surfels.add_surfels(live_surfels)
            self.surfels.update_active_mask_gl()

            self._time += 1
            self.surfels.max_time = self._time
            self._update_pose_indexmap(proj_matrix, frame_pcl.kcam, rt_cam,
                                       width, height)
            return FusionStats(live_surfels.size, 0, 0)

        indexmap_size = int(
            width*self.indexmap_scale), int(height*self.indexmap_scale)

        self.model_indexmap.raster(proj_matrix, rt_cam,
                                   indexmap_size[0], indexmap_size[1])
        live_query = self.live_merge_map.find_mergeable(
            self.model_indexmap, live_surfels,
            proj_matrix, width, height)
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
        self._merge_live_surfels(
            live_surfels.index_select(live_idxs),
            model_surfels, model_idxs.to(device))

        active_count = self.surfels.num_active_surfels()
        self._remove_surfels(visible_model_idxs)
        self.surfels.update_active_mask_gl()

        self.model_indexmap.raster(
            proj_matrix, rt_cam, indexmap_size[0], indexmap_size[1])
        dest_idxs, merge_idxs = self.intra_merge_map.find_mergeable_surfels(
            self.model_indexmap,
            self.stable_conf_thresh)
        self._merge_intra_surfels(dest_idxs, merge_idxs)

        # self.model_indexmap.raster(proj_matrix, rt_cam,
        #  indexmap_size[0], indexmap_size[1])
        self.spacecarving.carve(self.model_indexmap, proj_matrix, rt_cam, width,
                                height, self._time)

        self.surfels.update_active_mask_gl()

        removed_count = active_count - self.surfels.num_active_surfels()
        self._time += 1
        self.surfels.max_time = self._time

        self.surfels.update_active_mask_gl()
        self._update_pose_indexmap(
            proj_matrix, frame_pcl.kcam, rt_cam, width, height)
        return FusionStats(live_unst_idxs.size(0), model_idxs.size(0),
                           removed_count)

    def _merge_live_surfels(self, live, model, model_idxs):
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

    def _merge_intra_surfels(self, dest_idxs, merge_idxs):
        with self.surfels.context.current():
            with self.surfels.confs.as_tensor() as confs:
                dest_confs = confs[dest_idxs]
                merge_confs = confs[merge_idxs]
                confs_update = dest_confs + merge_confs

                with self.surfels.points.as_tensor() as points:
                    points[dest_idxs] = (points[dest_idxs]*dest_confs
                                         + points[merge_idxs]*merge_confs) / confs_update

                with self.surfels.colors.as_tensor() as colors:
                    updt_colors = (colors[dest_idxs].float()*dest_confs
                                   + colors[merge_idxs].float()*merge_confs) / confs_update
                    colors[dest_idxs] = updt_colors.byte()

                with self.surfels.normals.as_tensor() as normals:
                    normals[dest_idxs] = (normals[dest_idxs]*dest_confs
                                          + normals[merge_idxs]*merge_confs) / confs_update

                confs[dest_idxs] = confs_update

        self.surfels.mark_inactive(merge_idxs)

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

    def _update_pose_indexmap(self, proj_matrix, kcam, rt_cam, width, height):
        self.pose_indexmap.raster(proj_matrix, rt_cam, width, height, -1.0, -1)
        self._last_kcam = kcam
        self._last_rtcam = rt_cam

    def get_last_view_frame_pcl(self):
        with self.surfels.context.current():
            mask = self.pose_indexmap.index_tex.to_tensor()[:, :, 1]
            points = self.pose_indexmap.position_confidence_tex.to_tensor()[
                :, :, :3]
            normals = self.pose_indexmap.normal_radius_tex.to_tensor()[
                :, :, :3]
            colors = self.pose_indexmap.color_tex.to_tensor()

        return fiontb.frame.FramePointCloud(
            None, mask.byte(), self._last_kcam, self._last_rtcam,
            points, normals, colors)
