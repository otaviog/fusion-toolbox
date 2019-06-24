from pathlib import Path

import torch
import matplotlib.pyplot as plt

from ._raster import LiveSurfelRaster, GlobalSurfelRaster
from ._ckernels import surfel_find_live_to_model_merges

_SHADER_DIR = Path(__file__).parent / "shaders"


class IndexMap:
    def __init__(self, surfel_model):
        self.live_raster = LiveSurfelRaster(surfel_model.context)
        self.model_raster = GlobalSurfelRaster(surfel_model)

    def find_mergeable(self, surfel_cloud, proj_matrix, rt_cam,
                       width, height, debug=False):
        self.model_raster.raster(
            proj_matrix, rt_cam, width*4, height*4, -1.0, -1)
        self.model_raster.show_debug("Model", debug)

        self.live_raster.raster(surfel_cloud, proj_matrix, width, height)
        self.live_raster.show_debug("Live", debug)

        context = self.live_raster.context
        with context.current():
            live_pos = self.live_raster.pos_tex.to_tensor()
            live_normals = self.live_raster.normal_rad_tex.to_tensor()
            live_idxs = self.live_raster.idx_tex.to_tensor()

            model_pos = self.model_raster.pos_tex.to_tensor()
            model_normals = self.model_raster.normal_rad_tex.to_tensor()
            model_idxs = self.model_raster.idx_tex.to_tensor()

        _SENTINEL_TENSOR = torch.tensor([])
        live_features = model_features = _SENTINEL_TENSOR
        
        if surfel_cloud.features is not None:
            live_features = surfel_cloud.features
            model_features = self.model_raster.surfel_model.features
        
        merge_map = surfel_find_live_to_model_merges(
            live_pos, live_normals, live_idxs, live_features,
            model_pos, model_normals, model_idxs, model_features,
            0.5, False)

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

        visible_model_idxs = model_idxs[model_idxs[:, :, 1].byte()][:, 0].long()

        return (frame_fuse_idxs, model_fuse_idxs, frame_unstable_idxs,
                visible_model_idxs)

def _test():
    import cv2
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud
    from fiontb.frame import Frame, FrameInfo, FramePointCloud

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
    indexmap = IndexMap(surfel_model)

    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    frame_pcl = FramePointCloud(frame)
    surfel_cloud = SurfelCloud.from_frame_pcl(frame_pcl, 0, device="cuda:0")

    indexmap.find_mergeable(surfel_cloud, proj_matrix,
                            rt_cam, 640, 480, debug=True)


if __name__ == '__main__':
    _test()
