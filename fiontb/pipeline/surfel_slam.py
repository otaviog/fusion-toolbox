from fiontb.data.tumrgbd import read_trajectory
import math

import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.filtering import bilateral_depth_filter
from fiontb.camera import RTCamera
from fiontb.pose.icp import MultiscaleICPOdometry
from fiontb.pose import ICPVerifier
from fiontb.pose.autogradicp import MultiscaleAutogradICP
from fiontb.surfel import SurfelModel
from fiontb.fusion.surfel import SurfelFusion
from fiontb.filtering import BilateralDepthFilter


def _estimate_confidence_weight(prev_rt_cam, curr_rt_cam):
    pose = curr_rt_cam.difference(prev_rt_cam)
    angular_vel = pose.rodrigues().norm().item()
    pos_vel = pose.translation().norm().item()

    conf_weight = max(angular_vel, pos_vel)
    conf_weight = min(conf_weight, 0.01)
    conf_weight = max(1 - (conf_weight / 0.01), 0.5) * 1
    return conf_weight


_confs = [0.926322,
          0.691184,
          0.82881,
          0.848098,
          0.870813,
          0.817302,
          0.5,
          0.5,
          0.658203,
          0.666729,
          0.781634,
          0.723786,
          0.668831,
          0.857454,
          0.830854,
          0.654733,
          0.755859,
          0.695068,
          0.674766,
          0.830854,
          0.767842,
          0.719504,
          0.711129,
          0.514166,
          0.609375,
          0.5,
          0.5,
          0.647895,
          0.5,
          0.5,
          0.6339,
          0.641188,
          0.563268,
          0.767115,
          0.609735,
          0.70299,
          0.765276,
          0.5,
          0.566006,
          0.634604,
          0.5,
          0.55443,
          0.670357,
          0.640354,
          0.61864,
          0.547587,
          0.5,
          0.5,
          0.5,
          0.591474,
          0.574326,
          0.5,
          0.5,
          0.615089,
          0.536775]


gt_traj = read_trajectory(
    '/home/otaviog/3drec/slam-feature/exps/slam/045.klg.freiburg')
_cams = list(gt_traj.values())


class SurfelSLAM:
    def __init__(self, max_surfels=1024*1024*15, device="cuda:0",
                 tracking='frame-to-frame',
                 max_merge_distance=0.001,
                 normal_max_angle=math.radians(30),
                 stable_conf_thresh=10,
                 max_unstable_time=20, search_size=4, indexmap_scale=4,
                 min_z_difference=0.5, feature_size=3):
        self.device = device

        self.previous_fpcl = None
        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry(
            [(1.0, 30, True),
             (0.5, 30, True),
             (0.5, 30, True)])
        self.icp_verifier = ICPVerifier()

        self.previous_fpcl = None
        self._previous_features = None

        self._prev_frame_fpcl = None
        self._prev_frame_features = None

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

        confidence_weight = 1.0
        if self.previous_fpcl is not None:
            if self.fusion._time == 344:
                import ipdb
                ipdb.set_trace()

            result = self.icp.estimate(
                frame.info.kcam, filtered_live_fpcl.points,
                filtered_live_fpcl.mask,
                source_feats=features,
                target_points=self.previous_fpcl.points,
                target_mask=self.previous_fpcl.mask,
                target_normals=self.previous_fpcl.normals,
                target_feats=self._previous_features,
                geom_weight=0, feat_weight=1)

            if self.icp_verifier(result):
                relative_cam = result.transform
            else:
                from fiontb.fusion.surfel.fusion import FusionStats
                return FusionStats()
                
                print("Tracking fail")

                result = self.icp.estimate(
                    frame.info.kcam, filtered_live_fpcl.points,
                    filtered_live_fpcl.mask,
                    source_feats=features,
                    target_points=self._prev_frame_fpcl.points,
                    target_mask=self._prev_frame_fpcl.mask,
                    target_normals=self._prev_frame_fpcl.normals,
                    target_feats=self._previous_features,
                    geom_weight=0, feat_weight=1)

                if not self.icp_verifier(result):
                    # if True:
                    # return FusionStats()
                    print("Tracking fail 2")
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.imshow(features.transpose(1, 0).transpose(1, 2).cpu())

                    plt.figure()
                    plt.imshow(self._previous_features.transpose(
                        1, 0).transpose(1, 2).cpu())

                    import ipdb
                    ipdb.set_trace()
                    # return FusionStats()

            rt_camera = self.rt_camera.integrate(relative_cam.cpu())
            confidence_weight = _estimate_confidence_weight(
                self.rt_camera, rt_camera)
            self.rt_camera = rt_camera
        live_fpcl.normals = filtered_live_fpcl.normals

        global _confs
        if len(_confs) > 0:
            confidence_weight = _confs[0]
            _confs = _confs[1:]
        if False:
            global _cams
            self.rt_camera = _cams[0]
            self.rt_camera.matrix = self.rt_camera.matrix.float()
            _cams = _cams[1:]
        print("Confidence: ", confidence_weight)
        stats = self.fusion.fuse(live_fpcl, self.rt_camera, features,
                                 confidence_weight=confidence_weight)

        self._prev_frame_fpcl = filtered_live_fpcl
        self._prev_frame_features = features

        if self.tracking == 'frame-to-frame' or self.model.max_time < 3:
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
                print("Not fill")
                self.previous_fpcl = filtered_live_fpcl
                self._previous_features = features

        return stats
