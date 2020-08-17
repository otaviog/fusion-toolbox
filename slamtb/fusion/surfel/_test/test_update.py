"""Tests only the update part of surfel fusion
"""

import math

import fire
import torch
import tenviz

from slamtb.data import set_start_at_eye
from slamtb.viz.surfelrender import show_surfels
from slamtb.surfel import SurfelModel, SurfelCloud
from slamtb.testing import load_sample2_dataset

from ..indexmap import ModelIndexMapRaster
from ..update import Update


class _Tests:
    @staticmethod
    def _test(elastic_fusion):
        dataset = set_start_at_eye(load_sample2_dataset())

        device = torch.device("cpu:0")
        gl_context = tenviz.Context()

        model = SurfelModel(gl_context, 640*480*2)

        frame0 = dataset[1]
        live_frame = dataset[14]

        model_surfels = SurfelCloud.from_frame(
            frame0, time=0, confidence_weight=0.8).to(device)
        model_surfels.itransform(frame0.info.rt_cam.cam_to_world.float())
        torch.manual_seed(10)
        #model_surfels.radii = torch.rand_like(model_surfels.radii)*0.0025 + 0.002
        #model_surfels.radii *= 0.25

        model.add_surfels(model_surfels, update_gl=True)

        live_surfels = SurfelCloud.from_frame(
            live_frame, time=1, confidence_weight=0.8).to(device)
        #live_surfels.radii *= 0.25

        height, width = live_frame.depth_image.shape
        proj_matrix = live_frame.info.kcam.get_opengl_projection_matrix(
            0.01, 500.0)

        rt_cam = live_frame.info.rt_cam
        model_raster = ModelIndexMapRaster(model)
        model_raster.raster(
            proj_matrix, rt_cam, width, height)

        update = Update(elastic_fusion=elastic_fusion,
                        search_size=2, max_normal_angle=math.radians(30))
        prev_model = model.clone()

        with gl_context.current():
            model_indexmap = model_raster.to_indexmap(device)

        new_surfels = update(
            model_indexmap, live_surfels, live_frame.info.kcam,
            rt_cam, 1, model)
        model.add_surfels(new_surfels.to("cuda:0"), update_gl=True)
        print("Add {} new surfels".format(new_surfels.size))
        print("Merged {} surfels".format(live_surfels.size - new_surfels.size))
        show_surfels(gl_context,
                     [prev_model, live_surfels.transform(rt_cam.cam_to_world.float()),
                      model],
                     title="Update test", invert_y=True)

    def vanilla(self):
        """Test the update of our version.
        """
        self._test(False)

    def ef_like(self):
        """Test the update of elastic fusion.
        """
        self._test(True)


if __name__ == '__main__':
    fire.Fire(_Tests)
