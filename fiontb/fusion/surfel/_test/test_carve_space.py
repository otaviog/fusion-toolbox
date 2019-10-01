from pathlib import Path

import torch
import numpy as np
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelModel, SurfelCloud


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample2")
    set_cameras_to_start_at_eye(dataset)

    device = torch.device("cpu:0")
    gl_context = tenviz.Context()

    frame = dataset[0]
    surfel_cloud = SurfelCloud.from_frame(dataset[0])
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
    surfel_model.add_surfels(violations, update_gl=True)

    show_surfels(gl_context, [surfel_model])


if __name__ == '__main__':
    _test()
