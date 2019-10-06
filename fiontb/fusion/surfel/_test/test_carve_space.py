from pathlib import Path

import torch
import numpy as np
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster
from ..carve_space import CarveSpace

_STABLE_CONF_THRESH = 10.0
_CURRENT_TIME = 5


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample2")
    set_cameras_to_start_at_eye(dataset)

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
    violations_sampling = np.random.choice(surfel_model.allocator.active_count,
                                           num_violations)
    violations = surfel_cloud[violations_sampling]
    violations.positions += (frame.info.rt_cam.center - violations.positions)\
        * torch.rand(num_violations, 1)*0.8
    violations.times[:] = _CURRENT_TIME - 1
    violations.confidences[:] = _STABLE_CONF_THRESH - 1
    surfel_model.add_surfels(violations, update_gl=True)

    prev_model = surfel_model.clone()

    carve = CarveSpace(_STABLE_CONF_THRESH, search_size=2, min_z_difference=0.5)

    height, width = frame.depth_image.shape

    raster = ModelIndexMapRaster(surfel_model)
    raster.raster(frame.info.kcam.get_opengl_projection_matrix(0.01, 500.0),
                  frame.info.rt_cam, width, height)
    with gl_context.current():
        indexmap = raster.to_indexmap(device)

    carve(indexmap, _CURRENT_TIME, surfel_model, update_gl=True)

    show_surfels(gl_context, [prev_model, surfel_model])


if __name__ == '__main__':
    _test()
