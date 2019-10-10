import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_depth_filter
from fiontb.camera import RTCamera
from fiontb.pose.icp import MultiscaleICPOdometry
from fiontb.pose.autogradicp import MultiscaleAutogradICP
from fiontb.surfel import SurfelModel
from fiontb.fusion.fsf import FSFFusion


class FSFSLAM:
    def __init__(self, max_local_surfels, max_local_frames, feature_size,
                 max_surfels=1024*1024*30, device="cuda:0",
                 tracking='frame-to-frame', stable_conf_thresh=10):
        self.device = device

        self.previous_fpcl = None
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry(
            [(0.25, 20, True),
             (0.5, 20, True),
             (1.0, 20, True)])

        self.icp1 = MultiscaleAutogradICP(
            [(0.25, 100, 20, False),
             (0.5, 100, 20, False),
             (1.0, 100, 20, False)])

        self.previous_fpcl = None
        self._previous_features = None
        self.gl_context = tenviz.Context()

        self.fusion = FSFFusion(self.gl_context, max_local_surfels, max_local_frames,
                                max_surfels, feature_size,
                                stable_conf_thresh=stable_conf_thresh)

        self.tracking = tracking

    @property
    def model(self):
        return self.fusion.global_model

    @property
    def stable_conf_thresh(self):
        return self.fusion.stable_conf_thresh

    def step(self, frame, features=None):
        device = self.device

        live_fpcl = FramePointCloud.from_frame(frame).to(device)

        filtered_depth = bilateral_depth_filter(
            torch.from_numpy(frame.depth_image).to(device),
            torch.from_numpy(frame.depth_image > 0).to(device),
            depth_scale=frame.info.depth_scale)
        frame.depth_image = filtered_depth
        filtered_live_fpcl = FramePointCloud.from_frame(frame).to(device)

        relative_cam = torch.eye(4, dtype=torch.float32)
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

            # self.rt_camera = self.rt_camera.integrate(relative_cam.cpu())

        live_fpcl.normals = filtered_live_fpcl.normals

        stats = self.fusion.fuse(live_fpcl, relative_cam, features)

        if self.tracking == 'frame-to-frame':
            self.previous_fpcl = filtered_live_fpcl
        elif self.tracking == 'frame-to-model':
            self.previous_fpcl = self.fusion.get_model_frame_pcl()

        self._previous_features = features
        return stats
