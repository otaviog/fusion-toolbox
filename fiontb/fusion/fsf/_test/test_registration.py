import open3d
import torch

import fiontb._cfiontb

from pathlib import Path
from cProfile import Profile


import tenviz


from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelCloud, SurfelModel
from fiontb.testing import prepare_frame
from fiontb._utils import profile

from ..global_registration import GlobalRegistration
from ..registration import SurfelCloudRegistration


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 20 frames
    set_cameras_to_start_at_eye(dataset)

    device = "cuda:0"

    scale = 1.5
    frame0, features0 = prepare_frame(
        dataset[0].clone(), scale=scale, filter_depth=True, compute_normals=True)
    frame1, features1 = prepare_frame(dataset[14].clone(), scale=scale, filter_depth=True,
                                      compute_normals=True)

    target_surfels = SurfelCloud.from_frame(frame0, time=0,
                                            features=features0).to(device)
    source_surfels = SurfelCloud.from_frame(frame1, time=0,
                                            features=features1).to(device)

    init_reg = GlobalRegistration()
    transform = init_reg.estimate(target_surfels, source_surfels)

    registration = SurfelCloudRegistration(100, 0.05)
    source_surfels.itransform(transform)
    with profile(Path(__file__).parent / "registration.prof"):
        transform = registration.estimate(target_surfels, source_surfels)
        source_surfels.itransform(transform)

    gl_context = tenviz.Context()

    show_surfels(gl_context, [SurfelModel.from_surfel_cloud(gl_context, target_surfels),
                              SurfelModel.from_surfel_cloud(gl_context, source_surfels)])


if __name__ == '__main__':
    _test()
