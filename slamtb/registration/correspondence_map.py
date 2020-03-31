import torch

from slamtb._cslamtb import (CorrespondenceMap as _CorrespondenceMap)


class CorrespondenceMap:
    def __init__(self, distance_threshold, normals_angle_thresh):
        self.distance_threshold = distance_threshold
        self.normals_angle_thresh = normals_angle_thresh

    def __call__(self, source_points, source_normals, source_mask, transform,
                 target_points, target_normals, target_mask, kcam):
        _CORRESP_INF = 0x0fffffff7f800000

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
            kcam, corresp_map, self.distance_threshold,
            self.normals_angle_thresh)

        return corresp_map
