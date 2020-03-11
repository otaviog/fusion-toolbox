"""Metrics that use polygon information.
"""


import scipy.spatial.ckdtree
import numpy as np
import torch


def reconstruction_accuracy(source_points, closest_pts, thresh_distance=0.01):
    distances = torch.norm(source_points - closest_pts, 2, dim=1)
    return torch.mean((distances < thresh_distance).float()).item()


def chamfer_score(source_points, gt_points, max_search_radius=4.0):
    src_kdtree = scipy.spatial.ckdtree.cKDTree(source_points, leafsize=128)
    dists, _ = src_kdtree.query_ball_point(gt_points, r=max_search_radius)
    return np.mean(dists)


def sample_points(verts, faces, point_density, normals=None):
    """Converts a mesh into a point-cloud.

    Args:

        verts (:obj:`numpy.ndarray`): Vertices array of shape [N, 3].

        faces (:obj:`numpy.ndarray`): Face indices array of shape [N 3].

        point_density (float): Ratio of points per face area.
    """

    # pylint: disable=invalid-name
    areas = np.empty((faces.shape[0], ))
    for i, (idx0, idx1, idx2) in enumerate(faces):
        p0, p1, p2 = verts[idx0], verts[idx1], verts[idx2]

        v0 = p1 - p0
        v1 = p2 - p0

        areas[i] = np.linalg.norm(np.cross(v0, v1), 2)*0.5

    total_area = areas.sum()

    points = []
    point_normals = []

    for i, (idx0, idx1, idx2) in enumerate(faces):
        p0, p1, p2 = verts[idx0], verts[idx1], verts[idx2]
        if normals is not None:
            n0, n1, n2 = normals[idx0], normals[idx1], normals[idx2]

        trig_area = areas[i]

        num_points = int((trig_area / total_area)*point_density)
        for _ in range(num_points):
            rand0 = np.random.random()
            rand1 = np.random.random()
            s0 = np.sqrt(rand0)

            rand_pt = p0*(1 - s0) + p1*(1 - rand1)*s0 + p2*rand1*s0
            points.append(rand_pt)
            if normals is not None:
                rand_norm = n0*(1 - s0) + n1*(1 - rand1)*s0 + n2*rand1*s0
                point_normals.append(rand_norm)

    return np.array(points), np.array(point_normals)
