from pathlib import Path

import torch
import numpy as np
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster
from ..remove_unstable import RemoveUnstable

_STABLE_CONF_THRESH = 10.0
_CURRENT_TIME = 5


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample2")
    set_cameras_to_start_at_eye(dataset)

    device = torch.device("cuda:0")
    gl_context = tenviz.Context()

    frame = dataset[0]
    surfel_cloud = SurfelCloud.from_frame(dataset[0], time=_CURRENT_TIME)
    surfel_cloud.confidences[:] = _STABLE_CONF_THRESH + 0.1

    torch.manual_seed(110)
    num_violations = 500
    violations_sampling = np.random.choice(surfel_cloud.size, num_violations)
    surfel_cloud.confidences[violations_sampling] = _STABLE_CONF_THRESH - 4
    surfel_cloud.times[violations_sampling] = _CURRENT_TIME - 5

    surfel_model = SurfelModel(gl_context, 640*480*2)
    surfel_model.add_surfels(surfel_cloud, update_gl=True)
    prev_model = surfel_model.clone()

    remove = RemoveUnstable(_STABLE_CONF_THRESH, 5)

    height, width = frame.depth_image.shape

    raster = ModelIndexMapRaster(surfel_model)
    raster.raster(frame.info.kcam.get_opengl_projection_matrix(0.01, 500.0),
                  frame.info.rt_cam, width, height)
    with gl_context.current():
        indexmap = raster.to_indexmap(device)

    remove(indexmap.indexmap, _CURRENT_TIME, surfel_model, update_gl=True)

    show_surfels(gl_context, [prev_model, surfel_model])


if __name__ == '__main__':
    _test()
