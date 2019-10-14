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
from fiontb.filtering import BilateralDepthFilter


class SurfelSLAM:
    def __init__(self, max_surfels=1024*1024*30, device="cuda:0",
                 tracking='frame-to-frame',
                 max_merge_distance=0.005,
                 normal_max_angle=math.radians(30),
                 stable_conf_thresh=10,
                 max_unstable_time=20, search_size=2, indexmap_scale=4,
                 min_z_difference=0.5, feature_size=3):
        self.device = device

        self.previous_fpcl = None
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry(
            [(0.25, 20, False),
             (0.5, 20, True),
             (1.0, 20, True)],
            lost_track_threshold=1e-1)

        self.previous_fpcl = None
        self._previous_features = None
        self.gl_context = tenviz.Context()

        self.model = SurfelModel(
            self.gl_context, max_surfels, feature_size=feature_size)
        self.fusion = SurfelFusion(self.model,
                                   max_merge_distance=max_merge_distance,
                                   normal_max_angle=normal_max_angle,
                                   stable_conf_thresh=stable_conf_thresh,
                                   max_unstable_time=max_unstable_time,
                                   search_size=search_size,
                                   indexmap_scale=indexmap_scale,
                                   min_z_difference=min_z_difference)

        self.tracking = tracking

        self._depth_filter = BilateralDepthFilter()

    def step(self, frame, features=None):
        device = self.device

        live_fpcl = FramePointCloud.from_frame(frame).to(device)
        if features is not None:
            features = features.to(device)

        filtered_depth = bilateral_depth_filter(
            torch.from_numpy(frame.depth_image).to(device),
            torch.from_numpy(frame.depth_image > 0).to(device),
            depth_scale=frame.info.depth_scale)
        frame.depth_image = filtered_depth
        filtered_live_fpcl = FramePointCloud.from_frame(frame).to(device)

        if self.previous_fpcl is not None:
            relative_cam, tracking_ok = self.icp.estimate(
                frame.info.kcam, filtered_live_fpcl.points,
                filtered_live_fpcl.mask,
                source_feats=features,
                target_points=self.previous_fpcl.points,
                target_mask=self.previous_fpcl.mask,
                target_normals=self.previous_fpcl.normals,
                target_feats=self._previous_features,
                geom_weight=.2, feat_weight=.8)
            print(tracking_ok)
            #if not tracking_ok:
            if False:
                from fiontb.fusion.surfel.fusion import FusionStats

                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(features.transpose(1, 0).transpose(1, 2).cpu())

                plt.figure()
                plt.imshow(self._previous_features.transpose(
                    1, 0).transpose(1, 2).cpu())

                import ipdb
                ipdb.set_trace()
                return FusionStats()

            self.rt_camera = self.rt_camera.integrate(relative_cam.cpu())

        live_fpcl.normals = filtered_live_fpcl.normals
        stats = self.fusion.fuse(live_fpcl, self.rt_camera, features)

        if self.tracking == 'frame-to-frame' or self.model.max_time < 12:
            self.previous_fpcl = filtered_live_fpcl
            self._previous_features = features
        elif self.tracking == 'frame-to-model':
            model_fpcl, model_features = self.fusion.get_model_frame_pcl()

            if model_fpcl is not None:
                if False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.imshow(frame.rgb_image)
                    plt.figure()
                    plt.imshow(model_fpcl.colors.cpu())

                    plt.figure()
                    plt.imshow(model_features.transpose(
                        1, 0).transpose(1, 2).cpu())
                    model_fpcl.plot_debug()

                model_fpcl.points[:, :, 2] = self._depth_filter(
                    model_fpcl.points[:, :, 2], model_fpcl.mask)
                self.previous_fpcl = model_fpcl
                self._previous_features = model_features
            else:
                self.previous_fpcl = filtered_live_fpcl
                self._previous_features = features

        return stats
