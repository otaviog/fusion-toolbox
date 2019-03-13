"""
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


def closest_points(source_points, verts, trigs, processes=4):
    cgal_trigs = []

    for i0, i1, i2 in trigs:
        v0 = Point_3(*verts[i0].tolist())
        v1 = Point_3(*verts[i1].tolist())
        v2 = Point_3(*verts[i2].tolist())

        cgal_trigs.append(Triangle_3(v0, v1, v2))

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
