import math

import torch
import tenviz

from fiontb.frame import FramePointCloud
from fiontb.processing import BilateralDepthFilter, bilateral_depth_filter
from fiontb.camera import RTCamera
from fiontb.pose.icp import MultiscaleICPOdometry, ICPOption, ICPVerifier
from fiontb.surfel import SurfelModel
from fiontb.fusion.surfel.effusion import EFFusion, FusionStats
from fiontb.fusion.surfel.indexmap import (ModelIndexMapRaster,
                                           SurfelIndexMapRaster)
from fiontb._cfiontb import SurfelFusionOp as _SurfelFusionOp


def _estimate_confidence_weight(prev_rt_cam, curr_rt_cam):
    pose = curr_rt_cam.difference(prev_rt_cam)
    angular_vel = pose.rodrigues().norm().item()
    pos_vel = pose.translation().norm().item()

    conf_weight = max(angular_vel, pos_vel)
    conf_weight = min(conf_weight, 0.01)
    conf_weight = max(1 - (conf_weight / 0.01), 0.5) * 1
    return conf_weight


_DEBUG = False

class SurfelSLAM:
    def __init__(self, max_surfels=1024*1024*15, device="cuda:0",
                 tracking_mode='frame-to-frame',
                 stable_conf_thresh=10,
                 stable_time_thresh=20, search_size=4, indexmap_scale=4,
                 feature_size=3):
        self.device = device

        self._depth_filter = BilateralDepthFilter()

        self.rt_camera = RTCamera(torch.eye(4, dtype=torch.float32))

        self.icp = MultiscaleICPOdometry([
            ICPOption(1.0, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            ICPOption(0.5, 10, geom_weight=10, feat_weight=1),
            # ICPOption(1.0, 10, feat_weight=1, so3=True),
        ])

        self.icp_verifier = ICPVerifier()

        self._previous_fpcl = None
        self._previous_features = None
        self._pose_raster = None

        self.gl_context = tenviz.Context()

        self.model = SurfelModel(
            self.gl_context, max_surfels, feature_size=feature_size)

        if tracking_mode == 'frame-to-model':
            self._pose_raster = SurfelIndexMapRaster(self.model)

        self.fusion = EFFusion(self.model,
                               stable_conf_thresh=stable_conf_thresh,
                               stable_time_thresh=stable_time_thresh,
                               search_size=search_size,
                               indexmap_scale=indexmap_scale)

    def step(self, frame, features=None):
        frame = frame.clone_shallow()
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
        if self._previous_fpcl is not None:
            previous_fpcl = self._previous_fpcl
            previous_features = self._previous_features

            if self._pose_raster is not None:
                model_fpcl, model_features = self._get_model_frame_pcl(
                    frame.info.kcam)

                if model_fpcl is not None:
                    previous_fpcl = model_fpcl
                    previous_features = model_features

            for _ in range(2 if self._pose_raster is not None else 1):
                previous_fpcl.points[:, :, 2] = self._depth_filter(
                    previous_fpcl.points[:, :, 2], previous_fpcl.mask)

                if _DEBUG:
                    import cv2
                    cv2.imshow("curr", cv2.cvtColor(filtered_live_fpcl.colors.cpu().numpy(),
                                                    cv2.COLOR_RGB2BGR))
                    cv2.imshow("prev", cv2.cvtColor(previous_fpcl.colors.cpu().numpy(),
                                                    cv2.COLOR_RGB2BGR))

                result = self.icp.estimate(
                    frame.info.kcam, filtered_live_fpcl.points,
                    filtered_live_fpcl.normals,
                    filtered_live_fpcl.mask,
                    source_feats=features,
                    target_points=previous_fpcl.points,
                    target_mask=previous_fpcl.mask,
                    target_normals=previous_fpcl.normals,
                    target_feats=previous_features)

                relative_cam = result.transform
                if self.icp_verifier(result):
                    break

                previous_fpcl = self._previous_fpcl
                previous_features = self._previous_features

            rt_camera = self.rt_camera.integrate(relative_cam.cpu())
            confidence_weight = _estimate_confidence_weight(
                self.rt_camera, rt_camera)
            self.rt_camera = rt_camera

        live_fpcl.normals = filtered_live_fpcl.normals

        stats = self.fusion.fuse(live_fpcl, self.rt_camera, features,
                                 confidence_weight=confidence_weight)

        self._previous_fpcl = live_fpcl
        self._previous_features = features

        return stats

    def _get_model_frame_pcl(self, kcam, min_fill_ratio=0.75, flip=False):
        gl_proj_matrix = kcam.get_opengl_projection_matrix(
            0.01, 100.0, dtype=torch.float)
        self._pose_raster.raster(gl_proj_matrix, self.rt_camera,
                                 kcam.image_width, kcam.image_height,
                                 self.fusion.stable_conf_thresh)
        indexmap = self._pose_raster.to_indexmap()
        indexmap.synchronize()
        from ..fusion.surfel.indexmap import show_indexmap
        # show_indexmap(indexmap)
        import cv2
        cv2.imshow("model", cv2.cvtColor(
            indexmap.color.cpu().numpy(), cv2.COLOR_RGB2BGR))
        mask = indexmap.indexmap[:, :, 1]
        fill_ratio = mask.sum().item()/(mask.size(0)*mask.size(1))
        print(fill_ratio)
        if fill_ratio < min_fill_ratio:
            return None, None

        if flip:
            mask = mask.flip([0])
        else:
            mask = mask.clone()

        mask = mask.bool()
        features = None

        if self.model.has_features:
            features = torch.zeros(self.model.feature_size, mask.size(0), mask.size(1),
                                   device=self.model.device,
                                   dtype=torch.float)
            _SurfelFusionOp.copy_features(
                indexmap.indexmap, self.model.features, features, flip)

        if flip:
            points = indexmap.position_confidence[:, :, :3].clone().flip([0])
            normals = indexmap.normal_radius[:, :, :3].clone().flip([0])
            colors = indexmap.color.clone().flip([0])
        else:
            points = indexmap.position_confidence[:, :, :3].clone()
            normals = indexmap.normal_radius[:, :, :3].clone()
            colors = indexmap.color.clone()

        return FramePointCloud(
            None, mask, kcam, self.rt_camera,
            points, normals, colors), features
