"""
Multiscale optimzation shared functions.
"""

import torch

from fiontb.frame import Frame
from fiontb.processing import (
    downsample_xyz, downsample_mask, DownsampleXYZMethod, feature_pyramid)
from .result import ICPResult


class MultiscaleOptimization:
    def __init__(self, estimators,
                 downsample_xyz_method=DownsampleXYZMethod.Nearest):
        self.estimators = estimators
        self.downsample_xyz_method = downsample_xyz_method

    def estimate(self, kcam, source_points, source_normals, source_mask, source_feats=None,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, transform=None):
        """Estimate the ICP odometry between a target frame points and normals
        against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float (H x W x 3) tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float (H x W x 3) tensor
             of rasterized 3d normals that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A bool (H x W) mask tensor
             of valid target points.

            source_points (:obj:`torch.Tensor`): A float (H x W x 3) tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): A bool (H x W) mask tensor
             of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A (4 x 4) initial
             transformation matrix.

        Returns:

             (:obj:`ICPResult`): Resulting transformation and
              information.
        """

        has_feats = target_feats is not None and source_feats is not None

        tgt_feats = None
        src_feats = None

        pyramid = []

        for scale, _ in self.estimators:
            if scale < 1.0:
                target_points = downsample_xyz(target_points, target_mask, scale,
                                               method=self.downsample_xyz_method)
                target_normals = downsample_xyz(target_normals, target_mask, scale,
                                                normalize=True,
                                                method=self.downsample_xyz_method)

                target_mask = downsample_mask(target_mask, scale)

                source_points = downsample_xyz(source_points, source_mask, scale,
                                               normalize=False,
                                               method=self.downsample_xyz_method)
                source_normals = downsample_xyz(source_normals, source_mask, scale,
                                                normalize=True,
                                                method=self.downsample_xyz_method)

                source_mask = downsample_mask(source_mask, scale)

                if has_feats:
                    target_feats = torch.nn.functional.interpolate(
                        target_feats.unsqueeze(0), scale_factor=scale, mode='bilinear',
                        align_corners=False).squeeze(0)

                    source_feats = torch.nn.functional.interpolate(
                        source_feats.unsqueeze(0), scale_factor=scale, mode='bilinear',
                        align_corners=False).squeeze(0)

                kcam = kcam.scaled(scale)
                torch.cuda.synchronize()

            pyramid.append(
                (target_points, target_normals, target_mask,
                 source_points, source_normals, source_mask,
                 target_feats, source_feats, kcam))

        result = ICPResult()
        for icp_instance, (tgt_points, tgt_normals, tgt_mask,
                           src_points, src_normals, src_mask,
                           tgt_feats, src_feats, pyr_kcam) in zip(
                               self.estimators, pyramid[::-1]):
            icp_instance = icp_instance[1]

            curr_result = icp_instance.estimate(
                pyr_kcam, src_points, src_normals, src_mask,
                source_feats=src_feats,
                target_points=tgt_points,
                target_mask=tgt_mask, target_normals=tgt_normals,
                target_feats=tgt_feats,
                transform=transform)
            if curr_result:
                transform = curr_result.transform.clone()
                result = curr_result

        return result

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        from fiontb.frame import FramePointCloud

        if isinstance(source_frame, Frame):
            source_frame = FramePointCloud.from_frame(source_frame).to(device)

        if isinstance(target_frame, Frame):
            target_frame = FramePointCloud.from_frame(target_frame).to(device)

        scales = [scale for scale, _ in self.estimators]

        source_feat_pyr = target_feat_pyr = [None]*len(scales)

        source_pyr = source_frame.pyramid(scales)
        if source_feats is not None:
            source_feat_pyr = feature_pyramid(source_feats, scales)

        target_pyr = target_frame.pyramid(scales)
        if target_feats is not None:
            target_feat_pyr = feature_pyramid(target_feats, scales)

        for ((_, estimator), source, src_feats,
             target, tgt_feats) in zip(self.estimators[::-1],
                                       source_pyr, source_feat_pyr,
                                       target_pyr, target_feat_pyr):

            curr_result = estimator.estimate(
                source.kcam,
                source.points,
                source.normals,
                source.mask,
                source_feats=src_feats,
                target_points=target.points,
                target_normals=target.normals,
                target_mask=target.mask,
                target_feats=tgt_feats,
                transform=transform)

            if curr_result:
                transform = curr_result.transform
                result = curr_result

        return result
