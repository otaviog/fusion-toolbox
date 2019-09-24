import torch

from fiontb._utils import empty_ensured_size
from fiontb._cfiontb import Matching


class DensePointMatcher:
    def __init__(self):
        self.out_point = None
        self.out_index = None

    def match(self, target_points, target_mask, source_points, kcam, rt_cam):
        device = source_points.device
        self.out_point = empty_ensured_size(self.out_point, source_points.size(0), 3,
                                            dtype=source_points.dtype,
                                            device=device)
        self.out_index = empty_ensured_size(self.out_index, source_points.size(0),
                                            dtype=torch.int64, device=device)

        Matching.match_dense_points(target_points, target_mask, source_points,
                                    kcam.matrix, rt_cam,
                                    self.out_point, self.out_index)

        return self.out_point, self.out_index
