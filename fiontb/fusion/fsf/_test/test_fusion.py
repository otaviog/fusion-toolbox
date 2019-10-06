from pathlib import Path

import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.frame import FramePointCloud
from fiontb.viz.surfelrender import show_surfels
from fiontb.testing import prepare_frame

from ..fusion import FSFFusion


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 20 frames
    set_cameras_to_start_at_eye(dataset)

    gl_context = tenviz.Context()

    fusion = FSFFusion(gl_context, 1024*1024*3, 2, 1024*1024*24)
    for i in range(len(dataset)):
        frame = prepare_frame(dataset[i])

        frame_pcl = FramePointCloud.from_frame(frame)
        fusion.fuse(frame_pcl, frame_pcl.rt_cam)

    show_surfels(gl_context, [fusion.global_model])


if __name__ == '__main__':
    _test()
