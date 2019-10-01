from pathlib import Path
from cProfile import Profile

import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelCloud, SurfelModel
from fiontb.testing import prepare_frame

from ..registration import SurfelCloudRegistration



def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 20 frames
    set_cameras_to_start_at_eye(dataset)

    device = "cuda:0"

    target_surfels = SurfelCloud.from_frame(
        prepare_frame(dataset[0])).to(device)
    source_surfels = SurfelCloud.from_frame(
        prepare_frame(dataset[2])).to(device)

    registration = SurfelCloudRegistration(100, 0.05)

    prof = Profile()
    prof.enable()
    transform = registration.estimate(target_surfels, source_surfels)
    prof.disable()
    prof.dump_stats(str(Path(__file__).parent / "registration.prof"))
    source_surfels.itransform(transform)
    gl_context = tenviz.Context()

    show_surfels(gl_context, [SurfelModel.from_surfel_cloud(gl_context, target_surfels),
                              SurfelModel.from_surfel_cloud(gl_context, source_surfels)])


if __name__ == '__main__':
    _test()
