"""Sample slam using surfels
"""

import argparse

import torch
import onireader

import tenviz
from tenviz.io import write_3dobject

from fiontb.sensor import Sensor, PresetIntrinsics
from fiontb.camera import RTCamera
from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode
from fiontb.frame import FramePointCloud, estimate_normals
from fiontb.filtering import bilateral_filter_depth_image
from fiontb.pose.icp import ICPOdometry, MultiscaleICPOdometry
from fiontb.fusion.surfel import SurfelModel, SurfelFusion


def _main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.parse_args("")

    sensor_dev = onireader.Device()
    sensor_dev.open()
    #sensor_dev.start(*sensor_dev.find_best_fit_modes(640, 480))
    sensor_dev.start(4, 9)  # 640x480
    sensor = Sensor(sensor_dev, depth_cutoff=3.5*1000,
                    preset_intrinsics=PresetIntrinsics.ASUS_XTION)

    device = "cuda:0"
    context = tenviz.Context()
    surfel_model = SurfelModel(context, 1024*1024*10, device)
    fusion_ctx = SurfelFusion(surfel_model)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(surfel_model, RunMode.PLAY, inverse=True)

    rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
    prev_fpcl = None

    icp = MultiscaleICPOdometry([(0.25, 15), (0.5, 10), (1.0, 5)])
    # icp = ICPOdometry(20)
    for _ in rec_ui:
        frame = sensor.next_frame()
        if frame is None:
            break

        live_fpcl = FramePointCloud.from_frame(frame).to(device)

        filtered_depth_image = bilateral_filter_depth_image(
            torch.from_numpy(frame.depth_image).to(device),
            live_fpcl.mask, depth_scale=frame.info.depth_scale)

        live_fpcl = FramePointCloud.from_frame(frame).to(device)
        live_fpcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                             live_fpcl.mask)
        frame.normal_image = live_fpcl.normals.cpu()

        sensor_ui.update(frame)

        if prev_fpcl is not None:
            relative_cam = icp.estimate_frame_to_frame(
                prev_fpcl, live_fpcl)
        else:
            relative_cam = torch.eye(4, dtype=torch.float32)

        rt_cam = rt_cam.integrate(relative_cam.cpu())
        prev_fpcl = live_fpcl

        fusion_ctx.fuse(live_fpcl, rt_cam)


if __name__ == '__main__':
    _main()
