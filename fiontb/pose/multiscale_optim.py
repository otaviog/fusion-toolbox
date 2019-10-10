import torch

from fiontb.downsample import (
    downsample_xyz, downsample_mask, DownsampleXYZMethod)


class MultiscaleOptimization:
    def __init__(self, estimators,
                 downsample_xyz_method=DownsampleXYZMethod.Nearest):
        self.estimators = estimators
        self.downsample_xyz_method = downsample_xyz_method

    def estimate(self, kcam, source_points, source_mask, source_feats=None,
                 target_points=None, target_mask=None, target_normals=None,
                 target_feats=None, transform=None, geom_weight=1.0, feat_weight=1.0):
        """Estimate the ICP odometry between a target frame points and normals
        against source points using the point-to-plane geometric
        error.

        Args:

            target_points (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d points that the source points should be
             aligned.

            target_normals (:obj:`torch.Tensor`): A float [WxHx3] tensor
             of rasterized 3d normals that the source points should be
             aligned.

            target_mask (:obj:`torch.Tensor`): A uint8 [WxH] mask tensor
             of valid target points.

            source_points (:obj:`torch.Tensor`): A float [WxHx3] tensor of
             source points.

            source_mask (:obj:`torch.Tensor`): Uint8 [WxH] mask tensor
             of valid source points.

            kcam (:obj:`fiontb.camera.KCamera`): Intrinsics camera
             transformer.

            transform (:obj:`torch.Tensor`): A float [4x4] initial
             transformation matrix.

        Returns: (:obj:`torch.Tensor`): A [3x4] rigid motion matrix
            that aligns source points to target points.

        """

        has_feats = target_feats is not None and source_feats is not None

        tgt_feats = None
        src_feats = None
        for scale, icp_instance, use_feats in self.estimators:
            if scale < 1.0:
                tgt_points = downsample_xyz(target_points, target_mask, scale,
                                            method=self.downsample_xyz_method)
                tgt_normals = downsample_xyz(target_normals, target_mask, scale,
                                             normalize=True,
                                             method=self.downsample_xyz_method)

                tgt_mask = downsample_mask(target_mask, scale)

                src_points = downsample_xyz(source_points, source_mask, scale,
                                            normalize=False,
                                            method=self.downsample_xyz_method)
                src_mask = downsample_mask(source_mask, scale)

                if has_feats and use_feats:
                    tgt_feats = torch.nn.functional.interpolate(
                        target_feats.unsqueeze(0), scale_factor=scale, mode='bilinear',
                        align_corners=False)
                    tgt_feats = tgt_feats.squeeze(0)
                    src_feats = torch.nn.functional.interpolate(
                        source_feats.unsqueeze(0), scale_factor=scale, mode='bilinear',
                        align_corners=False)
                    src_feats = src_feats.squeeze(0)
            else:
                tgt_points = target_points
                tgt_normals = target_normals
                tgt_mask = target_mask

                src_points = source_points
                src_mask = source_mask

                if use_feats:
                    tgt_feats = target_feats
                    src_feats = source_feats

            scaled_kcam = kcam.scaled(scale)
            transform, tracking_ok = icp_instance.estimate(
                scaled_kcam, src_points, src_mask,
                source_feats=src_feats,
                target_points=tgt_points,
                target_mask=tgt_mask, target_normals=tgt_normals,
                target_feats=tgt_feats,
                transform=transform, geom_weight=geom_weight,
                feat_weight=feat_weight)

            if not tracking_ok:
                return transform, False
        return transform, True

    def estimate_frame_to_frame(self, target_frame, source_frame, transform=None):
        return self.estimate(target_frame.points, target_frame.normals,
                             target_frame.mask, source_frame.points, source_frame.mask,
                             source_frame.kcam, transform)
