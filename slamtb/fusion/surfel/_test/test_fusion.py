"""Test the fusion process using ground-truth camera poses.
"""

import argparse
from pathlib import Path
import math

import torch
import tenviz

from slamtb.data.ftb import load_ftb
from slamtb.data import set_start_at_eye
from slamtb.processing import bilateral_depth_filter
from slamtb.frame import FramePointCloud
from slamtb.surfel import SurfelModel
from slamtb.ui import FrameUI, SurfelReconstructionUI, RunMode
from slamtb.sensor import DatasetSensor
from slamtb._utils import profile

from slamtb.fusion.surfel.fusion import SurfelFusion
from slamtb.fusion.surfel.effusion import EFFusion


def _test():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Input FTB dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stable-conf-thresh", default=10.0, type=float)
    parser.add_argument("--stable-time-thresh", default=20, type=int)
    parser.add_argument("--search-size", default=2, type=int)
    parser.add_argument("--max-merge-distance", default=0.01, type=float)
    parser.add_argument("--depth-cutoff", type=float)
    parser.add_argument("--start-frame", default=1, type=int)
    parser.add_argument("--carve-z-tollerance", default=5e-2, type=float)
    parser.add_argument("--elastic-fusion", action="store_true")
    parser.add_argument("--title", default="Surfel fusion test")
    args = parser.parse_args()

    if args.dataset is None:
        test_data = Path(__file__).parent / "../../../../test-data/rgbd"

        dataset = load_ftb(test_data / "sample1")  # 30 frames
        dataset = set_start_at_eye(dataset)
    else:
        dataset = load_ftb(args.dataset)

    gl_context = tenviz.Context()

    model = SurfelModel(gl_context, 1024*1024*50)

    if not args.elastic_fusion:
        fusion = SurfelFusion(model,
                              normal_max_angle=math.radians(80),
                              stable_conf_thresh=args.stable_conf_thresh,
                              stable_time_thresh=args.stable_time_thresh,
                              search_size=args.search_size,
                              max_merge_distance=args.max_merge_distance,
                              carve_z_toll=args.carve_z_tollerance)
    else:
        fusion = EFFusion(model,
                          stable_conf_thresh=args.stable_conf_thresh,
                          stable_time_thresh=args.stable_time_thresh,
                          search_size=args.search_size)

    sensor_ui = FrameUI(args.title)
    rec_ui = SurfelReconstructionUI(model, RunMode.STEP,
                                    stable_conf_thresh=fusion.stable_conf_thresh,
                                    title=args.title)
    sensor = DatasetSensor(dataset, start_idx=args.start_frame,
                           depth_cutoff=args.depth_cutoff)
    device = args.device
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
            print("Frame {} - {}".format(sensor.current_idx, stats))


if __name__ == '__main__':
    _test()
