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
                 target_feats=None, transform=None):
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

            pyramid.append((target_points, target_normals, target_mask, source_points, source_mask,
                            target_feats, source_feats, kcam))

        for icp_instance, (tgt_points, tgt_normals, tgt_mask, src_points, src_mask,
                           tgt_feats, src_feats, pyr_kcam) in zip(self.estimators, pyramid[::-1]):
            icp_instance = icp_instance[1]

            result = icp_instance.estimate(
                pyr_kcam, src_points, src_mask,
                source_feats=src_feats,
                target_points=tgt_points,
                target_mask=tgt_mask, target_normals=tgt_normals,
                target_feats=tgt_feats,
                transform=transform)
            transform = result.transform

        return result

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        from fiontb.frame import FramePointCloud

        source_frame = FramePointCloud.from_frame(source_frame).to(device)
        target_frame = FramePointCloud.from_frame(target_frame).to(device)

        return self.estimate(source_frame.kcam, source_frame.points, source_frame.mask,
                             source_feats=source_feats,
                             target_points=target_frame.points,
                             target_mask=target_frame.mask,
                             target_normals=target_frame.normals,
                             target_feats=target_feats,
                             transform=transform)
