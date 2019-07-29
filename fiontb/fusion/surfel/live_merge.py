"""Auxiliary module for finding mergeable surfels.
"""

from pathlib import Path
import math

import torch
import matplotlib.pyplot as plt

from .indexmap import LiveIndexMap
from ._ckernels import surfel_find_live_to_model_merges, surfel_find_feat_live_to_model_merges

_SHADER_DIR = Path(__file__).parent / "shaders"


class LiveToModelMergeMap:
    def __init__(self, surfel_model,
                 normal_max_angle=math.radians(30),
                 search_size=2):
        self.surfel_model = surfel_model
        self.live_indexmap = LiveIndexMap(surfel_model.context)
        self.normal_max_angle = normal_max_angle
        self.search_size = search_size

    def find_mergeable(self, model_indexmap, live_surfels, proj_matrix,
                       width, height, debug=False):
        import ipdb; ipdb.set_trace()
        self.live_indexmap.raster(live_surfels, proj_matrix, width, height)
        self.live_indexmap.show_debug("Live", debug)

        context = self.live_indexmap.context
        with context.current():
            live_pos = self.live_indexmap.pos_tex.to_tensor()
            live_normals = self.live_indexmap.normal_rad_tex.to_tensor()
            live_idxs = self.live_indexmap.idx_tex.to_tensor()

            model_pos = model_indexmap.position_confidence_tex.to_tensor()
            model_normals = model_indexmap.normal_radius_tex.to_tensor()
            model_idxs = model_indexmap.index_tex.to_tensor()

        if live_surfels.features is None:
            merge_map = surfel_find_live_to_model_merges(
                live_pos, live_normals, live_idxs,
                model_pos, model_normals, model_idxs,
                self.normal_max_angle, self.search_size)
        else:
            live_features = live_surfels.features
            model_features = self.surfel_model.features
            merge_map = surfel_find_feat_live_to_model_merges(
                live_pos, live_normals, live_idxs, live_features,
                model_pos, model_normals, model_idxs, model_features,
                self.normal_max_angle, self.search_size)

        if debug:
            plt.figure("Merge map")
            plt.imshow(merge_map[:, :, 1].cpu())
            plt.show()

        write_mask = merge_map[:, :, 0] != 0

        model_merge_idxs = merge_map[:, :, 1][write_mask]
        frame_idxs = merge_map[:, :, 2][write_mask]

        fuse_mask = model_merge_idxs >= 0
        model_fuse_idxs = (model_merge_idxs[fuse_mask]).long()
        frame_fuse_idxs = frame_idxs[fuse_mask].long()
        frame_unstable_idxs = frame_idxs[model_merge_idxs == -1].long()

        visible_model_idxs = model_idxs[model_idxs[:,
                                                   :, 1].byte()][:, 0].long()

        return (frame_fuse_idxs, model_fuse_idxs, frame_unstable_idxs,
                visible_model_idxs)


def _test():
    import numpy as np
    import cv2
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.frame import Frame, FrameInfo, FramePointCloud
    from .model import SurfelModel, SurfelCloud
    from .indexmap import ModelIndexMap

    test_data = Path(__file__).parent / "_test"
    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))
    proj = tenviz.projection_from_kcam(
        kcam.matrix, 0.01, 10.0)
    proj_matrix = torch.from_numpy(proj.to_matrix()).float()

    rt_cam = RTCamera.create_from_pos_quat(
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681  # 72
    )

    with ctx.current():
        idxs = torch.arange(0, model_size, dtype=torch.long)
        surfel_model.points[idxs] = model.verts
        surfel_model.normals[idxs] = model.normals
        surfel_model.colors[idxs] = model.colors
        surfel_model.radii[idxs] = torch.full(
            (model_size, ), 0.05, dtype=torch.float)
        surfel_model.confs[idxs] = torch.full(
            (model_size, ), 2.1, dtype=torch.float)
        surfel_model.times[idxs] = torch.full(
            (model_size, ), 4, dtype=torch.int32)

    surfel_model.active_mask[idxs] = torch.zeros(
        (model_size,), dtype=torch.uint8, device="cuda:0")
    surfel_model.update_active_mask_gl()

    merge_map = LiveToModelMergeMap(surfel_model)
    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH).astype(np.int32)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    frame_pcl = FramePointCloud.from_frame(frame)
    live_surfels = SurfelCloud.from_frame_pcl(frame_pcl, 0, device="cuda:0")

    height, width = frame.depth_image.shape
    model_indexmap = ModelIndexMap(surfel_model)
    model_indexmap.raster(proj_matrix, rt_cam, width*4, height*4)
    model_indexmap.show_debug("Model")
    merge_map.find_mergeable(model_indexmap, live_surfels, proj_matrix,
                             width, height, debug=True)


if __name__ == '__main__':
    _test()
