"""Metrics for evaluation reconstruction quality.
"""

import numpy as np
import scipy.spatial.ckdtree

from .mesh import mesh_accuracy, sample_points
from .trajectory import absolute_translational_error, rotational_error, set_start_at_identity


def chamfer_score(source_points, gt_points):
    src_kdtree = scipy.spatial.ckdtree.cKDTree(source_points, leafsize=128)
    #dists, _ = src_kdtree.query(gt_points, k=1)
    dists, _ = src_kdtree.query_ball_point(gt_points, r=1.0)
    return np.mean(dists)
