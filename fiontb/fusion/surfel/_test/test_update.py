from pathlib import Path
import math

import fire
import torch
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster, show_indexmap
from ..update import Update


class Tests:
    def _test(self, elastic_fusion):
        test_data = Path(__file__).parent / "../../../../test-data/rgbd"

        dataset = set_start_at_eye(load_ftb(test_data / "sample2"))

        device = torch.device("cpu:0")
        gl_context = tenviz.Context()

        surfel_model = SurfelModel(gl_context, 640*480*2)

        model_surfels = SurfelCloud.from_frame(
            dataset[0], time=0, confidence_weight=0.8).to(device)
        torch.manual_seed(10)
        # model_surfels.radii = torch.rand_like(model_surfels.radii)*0.0025 + 0.002
        model_surfels.radii *= 0.25

        surfel_model.add_surfels(model_surfels, update_gl=True)

        live_frame = dataset[14]
        live_surfels = SurfelCloud.from_frame(
            live_frame, time=1, confidence_weight=0.8).to(device)
        live_surfels.radii *= 0.25

        height, width = live_frame.depth_image.shape
        proj_matrix = live_frame.info.kcam.get_opengl_projection_matrix(
            0.01, 500.0)

        rt_cam = live_frame.info.rt_cam
        model_raster = ModelIndexMapRaster(surfel_model)
        model_raster.raster(
            proj_matrix, rt_cam, width, height)

        update = Update(elastic_fusion=elastic_fusion,
                        search_size=2, max_normal_angle=math.radians(30))
        prev_model = surfel_model.clone()

        with gl_context.current():
            model_indexmap = model_raster.to_indexmap(device)

        new_surfels = update(
            model_indexmap, live_surfels, live_frame.info.kcam,
            rt_cam, 1, surfel_model)

        live_surfels.itransform(rt_cam.matrix)
        surfel_model.add_surfels(new_surfels.to("cuda:0"), update_gl=True)
        show_surfels(gl_context,
                     [prev_model,
                      SurfelModel.from_surfel_cloud(gl_context, live_surfels),
                      surfel_model])

    def vanilla(self):
        self._test(False)

    def ef_like(self):
        self._test(True)


if __name__ == '__main__':
    fire.Fire(Tests)
