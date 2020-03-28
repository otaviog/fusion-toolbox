"""
Multiscale optimzation shared functions.
"""


from slamtb.frame import Frame, FramePointCloud
from slamtb.processing import feature_pyramid, DownsampleXYZMethod


class MultiscaleRegistration:
    def __init__(self, estimators,
                 downsample_xyz_method=DownsampleXYZMethod.Nearest):
        """Setups the scale .

        Args:

            options (List[(float, obj)]): Each element contains a
             scale specific registraion estimator. Estimator should be specified
             with their scales from higher to lower. And they're applied
             from lower to higher.

            downsample_xyz_method
             (:obj:`slamtb.downsample.DownsampleXYZMethod`): Which
             method to interpolate the XYZ points and normals.

        """
        self.estimators = estimators
        self.downsample_xyz_method = downsample_xyz_method

    def estimate_frame(self, source_frame, target_frame, source_feats=None,
                       target_feats=None, transform=None, device="cpu"):
        if isinstance(source_frame, Frame):
            source_frame = FramePointCloud.from_frame(source_frame).to(device)

        if isinstance(target_frame, Frame):
            target_frame = FramePointCloud.from_frame(target_frame).to(device)

        scales = [scale for scale, _ in self.estimators]
        source_feat_pyr = target_feat_pyr = [None]*len(scales)

        source_pyr = source_frame.pyramid(
            scales, downsample_xyz_method=self.downsample_xyz_method, colored=False)
        if source_feats is not None:
            source_feat_pyr = feature_pyramid(source_feats, scales)

        target_pyr = target_frame.pyramid(
            scales,
            downsample_xyz_method=self.downsample_xyz_method, colored=False)
        if target_feats is not None:
            target_feat_pyr = feature_pyramid(target_feats, scales)

        for ((_, estimator), source, src_feats,
             target, tgt_feats) in zip(self.estimators[::-1],
                                       source_pyr, source_feat_pyr,
                                       target_pyr, target_feat_pyr):
            curr_result = estimator.estimate_frame(
                source,
                target,
                source_feats=src_feats,
                target_feats=tgt_feats,
                transform=transform,
                device=device)

            if curr_result:
                transform = curr_result.transform

        return curr_result

    def estimate_pcl(self, source_pcl, target_pcl, transform=None,
                     device="cpu"):
        """Registration for surfel or point clouds alignment.
        """

        if transform is not None:
            transform = transform.cpu()
        for scale, estimator in self.estimators[::-1]:
            if scale > 0:
                src = source_pcl.to("cpu").downsample(scale).to(device)
                tgt = target_pcl.to("cpu").downsample(scale).to(device)
            else:
                src = source_pcl
                tgt = target_pcl

            result = estimator.estimate_pcl(src, tgt, transform=transform,
                                            source_feats=src.features,
                                            target_feats=tgt.features)
            transform = result.transform

        return result
