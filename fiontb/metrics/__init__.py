"""Metrics for evaluation reconstruction quality.
"""

import numpy as np
from scipy.spatial.ckdtree import cKDTree

from .mesh import query_closest_points, mesh_accuracy, sample_points
from .trajectory import ate_rmse


def chamfer_score(source_points, gt_points):
    src_kdtree = cKDTree(source_points, leafsize=128)
    #dists, _ = src_kdtree.query(gt_points, k=1)
    dists, _ = src_kdtree.query_ball_point(gt_points, r=1.0)
    return np.mean(dists)
