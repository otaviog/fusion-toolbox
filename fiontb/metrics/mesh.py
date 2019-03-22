"""Metrics that use polygon information.
"""

from multiprocessing import Pool

import numpy as np

from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
from CGAL.CGAL_Kernel import Point_3, Triangle_3
from tqdm import tqdm


_TREE = None


def _process(point):
    point = Point_3(float(point[0]), float(point[1]), float(point[2]))
    closest = _TREE.closest_point(point)
    return closest.x(), closest.y(), closest.z()


def closest_points(source_points, verts, trigs, processes=2):
    """Returns the closest points on a mesh's surface from a given set of
    points.

    Args:

        source_points (:obj:`numpy.ndarray`): Source points, [Nx3] shape

        verts (:obj:`numpy.ndarray`): The mesh vertices, [Vx3] shape.

        faces (:obj:`numpy.ndarray`): The mesh face indices, [Fx3] shape.

    Returns: (:obj:`numpy.ndarray`): The closest points, [Nx3] shape.

    """

    cgal_trigs = []

    for i0, i1, i2 in tqdm(trigs, total=trigs.shape[0], desc="Loading CGAL trigs"):
        v0 = Point_3(*verts[i0].tolist())
        v1 = Point_3(*verts[i1].tolist())
        v2 = Point_3(*verts[i2].tolist())

        cgal_trigs.append(Triangle_3(v0, v1, v2))

    # pylint: disable=global-statement
    global _TREE
    _TREE = AABB_tree_Triangle_3_soup(cgal_trigs)

    result = np.empty_like(source_points)

    with Pool(processes=processes) as pool:
        for i, (x, y, z) in tqdm(enumerate(pool.map(_process, source_points)),
                                 total=len(source_points)):
            result[i, 0] = x
            result[i, 1] = y
            result[i, 2] = z

    return result


def mesh_accuracy(source_points, closest_pts, thresh_distance=0.01):
    distances = np.linalg.norm(source_points - closest_pts, 2, axis=1)

    return np.mean(distances < thresh_distance)


def sample_points(verts, faces, point_density):
    """Converts a mesh into a point-cloud.

    Args:

        verts (:obj:`numpy.ndarray`): Vertices array of shape [N, 3].

        faces (:obj:`numpy.ndarray`): Face indices array of shape [N 3].

        point_density (float): Ratio of points per face area.
    """

    areas = np.empty((faces.shape[0], ))
    for i, (idx0, idx1, idx2) in enumerate(faces):
        p0, p1, p2 = verts[idx0], verts[idx1], verts[idx2]

        v0 = p1 - p0
        v1 = p2 - p0

        areas[i] = np.linalg.norm(np.cross(v0, v1), 2)*0.5

    total_area = areas.sum()

    points = []
    for i, (idx0, idx1, idx2) in enumerate(faces):
        p0, p1, p2 = verts[idx0], verts[idx1], verts[idx2]
        trig_area = areas[i]

        num_points = int((trig_area / total_area)*point_density)
        for _ in range(num_points):
            rand0 = np.random.random()
            rand1 = np.random.random()
            s0 = np.sqrt(rand0)

            rand_pt = p0*(1 - s0) + p1*(1 - rand1)*s0 + p2*rand1*s0
            points.append(rand_pt)

    return np.array(points)
