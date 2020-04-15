import torch

from slamtb._cslamtb import (CorrespondenceMap as _CorrespondenceMap)

_CORRESP_INF = 0x0fffffff7f800000


class CorrespondenceMap:
    """Given a point-cloud arranged as an image, computes map where the
    elements are index to the nearest point in relation to another
    point-cloud (source). The nearest point are compuated by
    projecting the source points into the target point image.

    Attributes:

        distance_threshold (float): Maximum distance threshold.

        normals_angle_thresh (float): Maximum normal angle (in
        radians) between the target's normals and the source one.

    """

    def __init__(self, distance_threshold, normals_angle_thresh):
        self.distance_threshold = distance_threshold
        self.normals_angle_thresh = normals_angle_thresh

    def __call__(self, source_points, source_normals, source_mask, transform,
                 target_points, target_normals, target_mask, kcam):
        """Compute the correspondence map.

        Args:

            source_points (:obj:`torch.Tensor`): Source point-cloud
             points. Tensor with shape [Nx3].

            source_normals (:obj:`torch.Tensor`): Source point-cloud
             normals. Tensor with shape [Nx3].

            source_mask (:obj:`torch.Tensor`): Valid source point
             positions. Bool tensor with shape [N] or [1] to disable it.

            transform (:obj:`torch.Tensor`): Rigid
             transformation. Tensor with shape [4x4] or [3x4] on cpu.

            target_points (:obj:`torch.Tensor`): Target point-cloud
             points. Tensor with shape [HxWx3].

            target_normals (:obj:`torch.Tensor`): Target point-cloud
             normals. Tensor with shape [HxWx3].

            target_mask (:obj:`torch.Tensor`): Target point-cloud positions. Bool tensor with
             shape [HxW].

            kcam (:obj:`torch.Tensor`): Intrinsics camera
             matrix. Tensor with shape [3x3] or [2x3].

        Returns: (:obj:`torch.Tensor`):

            The correspondence map. Each element is an index pointing
             to the source point-cloud that corresponds as the nearest
             at that position. Tensor of shape [HxW] of type
             int64. The first 32-bits are float32 distance value, and
             the rest is point index.
        """
        device = source_points.device
        corresp_map = torch.full((target_points.size(0),
                                  target_points.size(1)),
                                 _CORRESP_INF,
                                 dtype=torch.int64,
                                 device=device)

        _CorrespondenceMap.compute_correspondence_map(
            source_points.view(-1, 3), source_normals.view(-1, 3),
            source_mask.view(-1), transform,
            target_points, target_normals, target_mask,
            kcam, corresp_map, float(self.distance_threshold),
            float(self.normals_angle_thresh))

        return corresp_map
