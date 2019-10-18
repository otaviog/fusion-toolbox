from pathlib import Path
import math

import torch
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster, LiveIndexMapRaster, show_indexmap
from ..merge_live import MergeLiveSurfels


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample2")
    set_cameras_to_start_at_eye(dataset)

    device = torch.device("cpu:0")
    gl_context = tenviz.Context()

    surfel_model = SurfelModel(gl_context, 640*480*2)

    model_surfels = SurfelCloud.from_frame(dataset[0])
    torch.manual_seed(10)
    model_surfels.radii = torch.rand_like(model_surfels.radii)*0.0025 + 0.002

    surfel_model.add_surfels(model_surfels, update_gl=True)

    live_frame = dataset[14]
    live_surfels = SurfelCloud.from_frame(live_frame)
    live_surfels.radii *= 0.25

    height, width = live_frame.depth_image.shape
    proj_matrix = live_frame.info.kcam.get_opengl_projection_matrix(
        0.01, 500.0)

    rt_cam = live_frame.info.rt_cam
    model_raster = ModelIndexMapRaster(surfel_model)
    model_raster.raster(
        proj_matrix, rt_cam, width, height)

    merge_live = MergeLiveSurfels(gl_context, 2, math.radians(30))
    prev_model = surfel_model.clone()

    with gl_context.current():
        model_indexmap = model_raster.to_indexmap(device)

    new_surfels = merge_live(
        model_indexmap, live_surfels, proj_matrix,
        live_frame.info.rt_cam, width, height,
        surfel_model)

    live_surfels.itransform(rt_cam.matrix)
    surfel_model.add_surfels(new_surfels.to("cuda:0"), update_gl=True)
    show_surfels(gl_context, [prev_model, SurfelModel.from_surfel_cloud(gl_context, live_surfels),
                              surfel_model])


if __name__ == '__main__':
    _test()
