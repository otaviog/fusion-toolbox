"""Sample slam using surfels
"""

import argparse

import torch
import onireader

import tenviz

from slamtb.sensor import Sensor, PresetIntrinsics, DatasetSensor
from slamtb.data.ftb import load_ftb
from slamtb.camera import RTCamera
from slamtb.ui import FrameUI, SurfelReconstructionUI, RunMode
from slamtb.frame import FramePointCloud
from slamtb.processing import bilateral_depth_filter, estimate_normals
from slamtb.registration.icp import (MultiscaleICPOdometry, ICPOptions)
from slamtb.registration.result import RegistrationVerifier

from slamtb.surfel import SurfelModel
from slamtb.fusion.surfel import SurfelFusion


def _main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument(
        "--ftb",
        help="Instead of using a RGBD device, read the data from a ftb dataset directory")
    args = arg_parser.parse_args()

    if args.ftb is None:
        sensor_dev = onireader.Device()
        sensor_dev.open()
        sensor_dev.start_nearest_vmode(640, 480)

        sensor = Sensor(sensor_dev, depth_cutoff=3.5*1000,
                        preset_intrinsics=PresetIntrinsics.ASUS_XTION)
    else:
        dataset = load_ftb(args.ftb)
        sensor = DatasetSensor(dataset)

    icp = MultiscaleICPOdometry([ICPOptions(1.0, 15),
                                 ICPOptions(0.5, 10),
                                 ICPOptions(0.25, 5)])
    icp_verifier = RegistrationVerifier()

    device = "cuda:0"
    context = tenviz.Context()

    surfel_model = SurfelModel(context, 1024*1024*10, device)
    fusion_ctx = SurfelFusion(surfel_model)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(surfel_model, RunMode.PLAY, inverse=True)

    rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
    prev_fpcl = None

    for _ in rec_ui:
        frame = sensor.next_frame()
        if frame is None:
            break

        live_fpcl = FramePointCloud.from_frame(frame).to(device)

        filtered_depth_image = bilateral_depth_filter(
            torch.from_numpy(frame.depth_image).to(device),
            live_fpcl.mask, depth_scale=frame.info.depth_scale)

        live_fpcl = FramePointCloud.from_frame(frame).to(device)
        live_fpcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                             live_fpcl.mask)
        frame.normal_image = live_fpcl.normals.cpu()

        sensor_ui.update(frame)

        if prev_fpcl is not None:
            icp_result = icp.estimate_frame(
                live_fpcl, target_fpcl)
            if not icp_verifier(icp_result):
                print("Tracking error")
            relative_transform = icp_result.transform
        else:
            relative_transform = torch.eye(4, dtype=torch.double)

        rt_cam = rt_cam.transform(relative_transform)
        prev_fpcl = live_fpcl

        fusion_ctx.fuse(live_fpcl, rt_cam)


if __name__ == '__main__':
    _main()
