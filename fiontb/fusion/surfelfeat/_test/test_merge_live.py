from pathlib import Path
import math

import torch
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.frame import FramePointCloud
from fiontb._cfiontb import FeatSurfel
from fiontb.viz.surfelrender import show_surfels

from ..indexmap import ModelIndexMapRaster, LiveIndexMapRaster, show_indexmap
from ..merge_live import MergeLive
from ..context import SurfelModel, LiveSurfels


def _test():
    test_data = Path(__file__).parent / "../../../pose/_test"

    dataset = load_ftb(test_data / "sample1")

    model_pcl = FramePointCloud.from_frame(
        dataset[0]).unordered_point_cloud(world_space=False)

    device = torch.device("cpu:0")
    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_pcl.size * 2, device)

    with ctx.current():
        idxs = torch.arange(0, model_pcl.size, dtype=torch.long)
        surfel_model.positions[idxs] = model_pcl.points
        surfel_model.confidences[idxs] = torch.full(
            (model_pcl.size, ), 2.1, dtype=torch.float)
        surfel_model.normals[idxs] = model_pcl.normals
        surfel_model.radii[idxs] = torch.full(
            (model_pcl.size, ), 0.005, dtype=torch.float)
        surfel_model.colors[idxs] = model_pcl.colors

    surfel_model.active_mask[idxs] = torch.zeros(
        (model_pcl.size,), dtype=torch.uint8, device=device)
    surfel_model.update_gl()

    live_frame = dataset[1]
    live_surfels = LiveSurfels.from_frame_pcl(
        FramePointCloud.from_frame(live_frame))

    height, width = live_frame.depth_image.shape
    proj_matrix = live_frame.info.kcam.get_opengl_projection_matrix(0.01, 10.0)

    model_raster = ModelIndexMapRaster(surfel_model)
    model_raster.raster(
        proj_matrix, live_frame.info.rt_cam, width*4, height*4)

    live_raster = LiveIndexMapRaster(ctx)
    live_raster.raster(live_surfels, proj_matrix, width, height)

    merge_live = MergeLive(2, math.radians(30))

    prev_model = surfel_model.clone()
    with ctx.current():
        model_indexmap = model_raster.get(device)
        live_indexmap = live_raster.get(device)

        show_indexmap(model_indexmap, "Model")
        # show_indexmap(live_indexmap, "Live")

        with surfel_model.map_as_tensors(device) as mapped_model:
            new_surfels_map = merge_live(
                model_indexmap, live_indexmap, mapped_model)

    new_surfels_index = new_surfels_map[new_surfels_map > -1]
    new_surfels = live_surfels[new_surfels_index].to("cuda:0")
    surfel_model.add_surfels(new_surfels, update_gl=True)

    show_surfels(ctx, [prev_model, surfel_model])


if __name__ == '__main__':
    _test()
