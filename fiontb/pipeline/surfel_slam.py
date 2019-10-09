import math

import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_depth_filter
from fiontb.camera import RTCamera
from fiontb.pose.icp import MultiscaleICPOdometry
from fiontb.pose.autogradicp import MultiscaleAutogradICP
from fiontb.surfel import SurfelModel
from fiontb.fusion.surfel import SurfelFusion


class SurfelSLAM:
    def __init__(self, max_surfels=1024*1024*30, device="cuda:0",
                 tracking='frame-to-frame',
                 max_merge_distance=0.005,
                 normal_max_angle=math.radians(30),
                 stable_conf_thresh=10,
                 max_unstable_time=20, search_size=2, indexmap_scale=4,
                 min_z_difference=0.5):
        self.device = device

        self.previous_fpcl = None
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry(
            [(0.25, 20, True),
             (0.5, 20, True),
             (1.0, 20, True)])

        self.icp2 = MultiscaleAutogradICP(
            [(0.25, 25, 0.05, True),
             (0.5, 25, 0.05, True),
             (1.0, 50, 0.05, True)])

        self.previous_fpcl = None
        self._previous_features = None
        self.gl_context = tenviz.Context()

        self.model = SurfelModel(self.gl_context, max_surfels)
        self.fusion = SurfelFusion(self.model,
                                   max_merge_distance=max_merge_distance,
                                   normal_max_angle=normal_max_angle,
                                   stable_conf_thresh=stable_conf_thresh,
                                   max_unstable_time=max_unstable_time,
                                   search_size=search_size,
                                   indexmap_scale=indexmap_scale,
                                   min_z_difference=min_z_difference)

        self.tracking = tracking

    def step(self, frame, features=None):
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
                source_feats=features,
                target_points=self.previous_fpcl.points,
                target_mask=self.previous_fpcl.mask,
                target_normals=self.previous_fpcl.normals,
                target_feats=self._previous_features,
                geom_weight=1.0, feat_weight=0.0)

            self.rt_camera = self.rt_camera.integrate(relative_cam.cpu())

        live_fpcl.normals = filtered_live_fpcl.normals
        stats = self.fusion.fuse(live_fpcl, self.rt_camera)

        if self.tracking == 'frame-to-frame':
            self.previous_fpcl = filtered_live_fpcl
        elif self.tracking == 'frame-to-model':
            model_fpcl = self.fusion.get_model_frame_pcl()

            self.previous_fpcl = (model_fpcl if model_fpcl is not None
                                  else filtered_live_fpcl)

        self._previous_features = features
        return stats
