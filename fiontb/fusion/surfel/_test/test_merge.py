from pathlib import Path

import torch
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster, show_indexmap
from ..merge import Merge


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample2")
    set_cameras_to_start_at_eye(dataset)

    device = torch.device("cuda:0")
    gl_context = tenviz.Context()

    surfel_model = SurfelModel(gl_context, 640*480*2)

    frame1 = dataset[1]
    surfel_model.add_surfels(
        SurfelCloud.from_frame(dataset[0]))

    surfel_model.add_surfels(
        SurfelCloud.from_frame(frame1).transform(
            frame1.info.rt_cam.cam_to_world),
        update_gl=True)

    height, width = frame1.depth_image.shape
    proj_matrix = frame1.info.kcam.get_opengl_projection_matrix(
        0.01, 500.0)

    rt_cam = frame1.info.rt_cam
    model_raster = ModelIndexMapRaster(surfel_model)
    model_raster.raster(
        proj_matrix, rt_cam, width, height)

    prev_model = surfel_model.clone()

    merge = Merge(max_distance=0.05, stable_conf_thresh=0.1)
    with gl_context.current():
        model_indexmap = model_raster.to_indexmap(device)

    # show_indexmap(model_indexmap, "model")
    merge_count = merge(model_indexmap, surfel_model, update_gl=True)
    print("Merged {}".format(merge_count))
    show_surfels(gl_context, [prev_model, surfel_model])


if __name__ == '__main__':
    _test()
