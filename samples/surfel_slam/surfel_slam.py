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
from slamtb.processing import (bilateral_depth_filter, estimate_normals,
                               to_color_feature, ColorSpace)
from slamtb.registration.preset import create_multiscale_odometry
from slamtb.registration import MultiscaleRegistration
from slamtb.registration.autogradicp import AutogradICP
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
        sensor = DatasetSensor(dataset, start_idx=66)

    iccp = create_multiscale_odometry()

    icp2 = MultiscaleRegistration([
        (1.0, AutogradICP(100, 0.05, geom_weight=1, feat_weight=1)),
        (0.5, AutogradICP(200, 0.05, geom_weight=1, feat_weight=1)),
        (0.5, AutogradICP(300, 0.05, geom_weight=1, feat_weight=1))])
    icp_verifier = RegistrationVerifier()

    device = "cuda:0"
    context = tenviz.Context()

    surfel_model = SurfelModel(context, 1024*1024*10, device)
    fusion_ctx = SurfelFusion(surfel_model)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(surfel_model, RunMode.STEP, inverse=True)

    rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
    prev_fpcl = None
    prev_features = None

    for frame_count, _ in enumerate(rec_ui):
        frame = sensor.next_frame()
        if frame is None:
            break

        live_fpcl = FramePointCloud.from_frame(frame).to(device)

        filtered_frame = frame.clone(shallow=True)
        filtered_frame.depth_image = bilateral_depth_filter(
            torch.from_numpy(frame.depth_image).to(device),
            live_fpcl.mask)

        filtered_live_fpcl = FramePointCloud.from_frame(
            filtered_frame).to(device)
        frame.normal_image = filtered_live_fpcl.normals.cpu().numpy()
        live_fpcl.normals = filtered_live_fpcl.normals
        sensor_ui.update(frame)

        features = to_color_feature(
            filtered_frame.rgb_image, ColorSpace.INTENSITY).to(device)
        if prev_fpcl is not None:
            icp_result = icp.estimate_frame(
                filtered_live_fpcl, prev_fpcl,
                source_feats=features,
                target_feats=prev_features)
            if not icp_verifier(icp_result):
                print("Tracking error at frame {}".format(frame_count))
            relative_transform = icp_result.transform
        else:
            relative_transform = torch.eye(4, dtype=torch.double)

        rt_cam = rt_cam.transform(relative_transform)
        prev_fpcl = filtered_live_fpcl
        prev_features = features

        fusion_ctx.fuse(live_fpcl, rt_cam)


if __name__ == '__main__':
    _main()
