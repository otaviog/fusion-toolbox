"""Metrics for evaluating geometry of reconstructions.
"""

import scipy.spatial.ckdtree
import numpy as np
import torch

from fiontb.spatial.trigoctree import TrigOctree


def mesh_reconstruction_accuracy(gt_verts, gt_trig_faces, pred_points,
                                 thresh_distance=0.01, octree_leaf_size=200):
    """Computes the reconstruction accuracy, or the percentage of points
    that are close to enough to its counter part on the ground truth.

    This function will create an Octree to speedup the computation of
    ground truth points on its mesh.

    Args:

        gt_verts (:obj:`torch.Tensor`): Ground truth mesh
         vertices. Float [Vx3] tensor.

        gt_trig_faces (:obj:`torch.Tensor`): Ground mesh triangle
         faces. Int64 [Fx3] tensor.

        pred_points (:obj:`torch.Tensor`): Predicted points. Float
         [Nx3] tensor.

        thresh_distance (float): Maximum distance between ground truth
         points and predicted ones for positive accuracy.

        octree_leaf_size (int): Number o triangles per Octree leaf.

Returns: (float): The ratio of points that are close to the ground
     truth.

    """
    octree = TrigOctree(gt_verts, gt_trig_faces.long(), octree_leaf_size)

    gt_closest, _ = octree.query_closest_points(pred_points)
    return reconstruction_accuracy(gt_closest, pred_points, thresh_distance)


def reconstruction_accuracy(closest_gt_points, pred_points, thresh_distance=0.01):
    """Computes the reconstruction accuracy, or the percentage of points
    that are close to enough to its counter part on the ground truth.

    This function expects that the closest ground truth points are
    already computed. See `mesh_reconstruction_accuracy` for a finding
    the those points.

    Args:

        closest_gt_points (:obj:`torch.Tensor`): Closest points on the
         ground truth. Float [Nx3] tensor.

        pred_points (:obj:`torch.Tensor`): Predicted points. Float [Nx3] tensor.

        thresh_distance (float): Maximum distance between ground truth
         points and predicted ones for positive accuracy.

    Returns: (float): The ratio of points that are close to the ground
     truth.
    """

    distances = torch.norm(closest_gt_points - pred_points, 2, dim=1)
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
