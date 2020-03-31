"""Utilities for the c++ class `slamtb::MergeMap`.
"""

import torch

_CORRESP_INF = 0x0fffffff7f800000


def create_merge_map(width, height, device):
    """Create a merge compatible with the C++ class (`slamtb::MergeMap`)
    """

    return torch.full((height, width),
                      _CORRESP_INF,
                      dtype=torch.int64,
                      device=device)
