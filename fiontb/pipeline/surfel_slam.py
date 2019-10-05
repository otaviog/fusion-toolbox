import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_depth_filter
from fiontb.camera import RTCamera
from fiontb.pose.icp import MultiscaleICPOdometry
from fiontb.surfel import SurfelModel
from fiontb.fusion.surfel import SurfelFusion

from fiontb.viz.show import show_pcls

class SurfelSLAM:
    def __init__(self, max_surfels=1024*1024*30, device="cuda:0"):
        self.device = device
        self.previous_fpcl = None
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry(
            [(0.25, 20, False),
             (0.5, 20, False),
             (1.0, 20, False)])

        self.previous_fpcl = None
        self.gl_context = tenviz.Context()

        self.model = SurfelModel(self.gl_context, max_surfels)
        self.fusion = SurfelFusion(self.model)

    def step(self, frame):
        device = self.device

        live_fpcl = FramePointCloud.from_frame(frame).to(device)

        filtered_depth = bilateral_depth_filter(
            torch.from_numpy(frame.depth_image).to(device),
            torch.from_numpy(frame.depth_image > 0).to(device),
            depth_scale=frame.info.depth_scale)
        frame.depth_image = filtered_depth
        filtered_live_fpcl = FramePointCloud.from_frame(frame).to(device)

        if self.previous_fpcl is not None:
            relative_cam = self.icp.estimate(
                frame.info.kcam, filtered_live_fpcl.points,
                filtered_live_fpcl.mask,
                target_points=self.previous_fpcl.points,
                target_mask=self.previous_fpcl.mask,
                target_normals=self.previous_fpcl.normals,
                geom_weight=1.0, feat_weight=0.0)

            self.rt_camera = self.rt_camera.integrate(relative_cam.cpu())

        stats = self.fusion.fuse(live_fpcl, self.rt_camera)
        self.previous_fpcl = filtered_live_fpcl

        return stats
