"""Tests only the cleaning of old surfels part from surfel fusion.
"""


import torch
import numpy as np
import tenviz


from slamtb.testing import load_sample2_dataset
from slamtb.data import set_start_at_eye
from slamtb.viz.surfelrender import show_surfels
from slamtb.surfel import SurfelModel, SurfelCloud

from ..indexmap import ModelIndexMapRaster
from ..clean import Clean

_STABLE_CONF_THRESH = 10.0
_CURRENT_TIME = 5


def _test():
    dataset = set_start_at_eye(load_sample2_dataset())

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

    clean = Clean(_STABLE_CONF_THRESH, 5)

    height, width = frame.depth_image.shape

    raster = ModelIndexMapRaster(surfel_model)
    raster.raster(frame.info.kcam.get_opengl_projection_matrix(0.01, 500.0),
                  frame.info.rt_cam, width, height)
    with gl_context.current():
        indexmap = raster.to_indexmap(device)

    clean(frame.info.kcam, frame.info.rt_cam,
          indexmap, _CURRENT_TIME, surfel_model, update_gl=True)

    show_surfels(gl_context, [prev_model, surfel_model])


if __name__ == '__main__':
    _test()
