"""Tests the surfel space carve algorithm.
"""
from pathlib import Path

from fire import Fire
import torch
import numpy as np
import tenviz

from slamtb.data.ftb import load_ftb
from slamtb.data import set_start_at_eye
from slamtb.viz.surfelrender import show_surfels
from slamtb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster
from ..carve_space import CarveSpace

_STABLE_CONF_THRESH = 10.0
_STABLE_TIME_THRESH = 20
_CURRENT_TIME = 30


def _test(z_toll=5e-2):
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = set_start_at_eye(load_ftb(test_data / "sample2"))

    device = torch.device("cpu:0")
    gl_context = tenviz.Context()

    frame = dataset[0]
    surfel_cloud = SurfelCloud.from_frame(dataset[0], time=_CURRENT_TIME)
    surfel_cloud.confidences[:] = _STABLE_CONF_THRESH + 0.1

    surfel_model = SurfelModel(gl_context, 640*480*2)
    surfel_model.add_surfels(surfel_cloud)

    np.random.seed(110)
    torch.manual_seed(110)

    num_violations = 500
    violations_sampling = np.random.choice(surfel_model.allocated_size,
                                           num_violations)
    violations = surfel_cloud[violations_sampling]
    violations.points += (torch.tensor(frame.info.rt_cam.center,
                                       dtype=torch.float32) - violations.points)\
        * torch.rand(num_violations, 1)*0.8
    violations.times[:] = _CURRENT_TIME - 1
    violations.confidences[:] = _STABLE_CONF_THRESH - 1
    surfel_model.add_surfels(violations, update_gl=True)

    prev_model = surfel_model.clone()

    carve = CarveSpace(_STABLE_CONF_THRESH, _STABLE_TIME_THRESH, search_size=2,
                       min_z_difference=z_toll)

    height, width = frame.depth_image.shape

    raster = ModelIndexMapRaster(surfel_model)
    raster.raster(frame.info.kcam.get_opengl_projection_matrix(0.01, 500.0),
                  frame.info.rt_cam, width, height)
    with gl_context.current():
        indexmap = raster.to_indexmap(device)

    carve(frame.info.kcam, frame.info.rt_cam,
          indexmap, _CURRENT_TIME, surfel_model, update_gl=True)

    print("1 - Before Space Carve")
    print("2 - After Space Carve")
    show_surfels(gl_context, [prev_model, surfel_model],
                 view_matrix=torch.tensor(
                     [[-0.994465, 0, -0.10507, 0.287367],
                      [-0.00702242, 0.997764, 0.0664658, -0.405826],
                      [0.104835, 0.0668358, -0.992241, -1.21483],
                      [0, 0, 0, 1]]))


if __name__ == '__main__':
    Fire(_test)
