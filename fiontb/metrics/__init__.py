"""Metrics for evaluation reconstruction quality.
"""

import numpy as np
import sklearn.neighbors


def chamfer_score(source_points, gt_points):
    src_kdtree = sklearn.neighbors.KDTree(source_points)
    dists, _ = src_kdtree.query(gt_points, k=1, dualtree=True)
    return np.mean(dists)
