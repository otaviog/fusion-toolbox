"""Sample slam using surfels
"""

import argparse

import torch
import onireader

import tenviz
from tenviz.io import write_3dobject

from fiontb.sensor import Sensor, DeviceType
from fiontb.camera import RTCamera
from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode
from fiontb.frame import FramePointCloud, estimate_normals
from fiontb.filtering import bilateral_filter_depth_image
from fiontb.pose.open3d import estimate_odometry
from fiontb.fusion.surfel import SurfelModel, SurfelFusion


def _main():
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.parse_args("")

    sensor_dev = onireader.Device()
    sensor_dev.open()
    sensor_dev.start(4, 12)

    sensor = Sensor(sensor_dev, DeviceType.ASUS_XTION, 3.5*1000)

    device = "cuda:0"
    context = tenviz.Context()
    surfel_model = SurfelModel(context, 1024*1024*10, device)
    fusion_ctx = SurfelFusion(surfel_model)

    sensor_ui = FrameUI("Frame Control")
    rec_ui = SurfelReconstructionUI(surfel_model, RunMode.PLAY, inverse=True)

    rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
    prev_frame = None

    for _ in rec_ui:
        frame = sensor.next_frame()
        if frame is None:
            break

        live_fpcl = FramePointCloud(frame)
        mask = live_fpcl.depth_mask.to(device)

        filtered_depth_image = bilateral_filter_depth_image(
            torch.from_numpy(frame.depth_image).to(device),
            mask, depth_scale=frame.info.depth_scale)

        live_fpcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                             mask).cpu()

        frame.normal_image = live_fpcl.normals

        sensor_ui.update(frame)

        if prev_frame is not None:
            relative_cam = estimate_odometry(
                prev_frame, frame)
        else:
            relative_cam = torch.eye(4, dtype=torch.float32)

        rt_cam = rt_cam.integrate(relative_cam)
        prev_frame = frame

        fusion_ctx.fuse(live_fpcl, rt_cam)


if __name__ == '__main__':
    _main()
