from pathlib import Path
import math

import torch
import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_start_at_eye
from fiontb.processing import bilateral_depth_filter
from fiontb.frame import FramePointCloud
from fiontb.surfel import SurfelModel
from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode
from fiontb.sensor import DatasetSensor
from fiontb._utils import profile

from ..fusion import SurfelFusion


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 30 frames
    dataset = set_start_at_eye(dataset)

    gl_context = tenviz.Context()

    model = SurfelModel(gl_context, 1024*1024*50)

    fusion = SurfelFusion(model, normal_max_angle=math.radians(
        80), max_merge_distance=0.5)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(model, RunMode.STEP,
                                    stable_conf_thresh=fusion.stable_conf_thresh)
    sensor = DatasetSensor(dataset)
    device = "cuda:0"
    with profile(Path(__file__).parent / "fusion.prof"):
        for _ in rec_ui:
            frame = sensor.next_frame()
            if frame is None:
                break

            sensor_ui.update(frame)

            fpcl = FramePointCloud.from_frame(frame).to(device)

            filtered_depth = bilateral_depth_filter(
                torch.from_numpy(frame.depth_image).to(device),
                torch.from_numpy(frame.depth_image > 0).to(device),
                depth_scale=frame.info.depth_scale)
            frame.depth_image = filtered_depth
            filtered_fpcl = FramePointCloud.from_frame(frame).to(device)
            fpcl.normals = filtered_fpcl.normals

            stats = fusion.fuse(fpcl, fpcl.rt_cam)
            print("Frame {} - {}".format(rec_ui.frame_count, stats))

if __name__ == '__main__':
    _test()
