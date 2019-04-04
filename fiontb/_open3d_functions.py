"""Utilities function to interact with open3d module.
"""

from open3d import PointCloud
import numpy as np

def from_open3d(pcl: PointCloud):
    """Converts from :class:`open3d.PointCloud` to a
    :class:`numpy.ndarray` used by fusionkit.

    >>> pcl = PointCloud()
    >>> pcl.points.append([1.0, 2.0, 3.0])
    >>> pcl.points.append([4.0, 5.0, 6.0])
    >>> pcl.points.append([7.0, 8.0, 9.0])
    >>> from_open3d(pcl)
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    """

    return np.asarray(pcl.points)


def to_open3d(points, colors=None, normals=None):
    """Converts fusionkit :class:`numpy.ndarray` of points to
    :class:`open3d.PointCloud`. Example:

    >>> fsi_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    >>> pcl = to_open3d(fsi_points)
    >>> pcl
    PointCloud with 3 points.
    >>> np.asarray(pcl.points)
    array([[1., 2., 3.],
           [4., 5., 6.],
           [7., 8., 9.]])
    """

    pcl = PointCloud()
    for point in points:
        pcl.points.append(point.squeeze())

    if colors is not None:
        for color in colors:
            pcl.colors.append(color)

    return pcl
