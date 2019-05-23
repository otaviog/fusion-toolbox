import numpy as np
import torch

from .camera import Homogeneous


class PointCloud:
    """Basic point cloud data

    Attributes:

        points (:obj:`numpy.ndarray`): Points [Nx3] of floats.

        colors (:obj:`numpy.ndarray`): Colors [Nx3] of uints8.

        normals (:obj:`numpy.ndarray`): Normals [Nx3] of floats.
    """

    @classmethod
    def from_open3d(cls, pcl):
        """Converts a :obj:`open3d.PointCloud` object to this project's Point
        Cloud.
        """
        return cls(np.array(pcl.points, dtype=np.float32), np.array(pcl.colors, dtype=np.uint8),
                   np.array(pcl.normals, dtype=np.float32))

    def __init__(self, points=np.array([], np.float32),
                 colors=np.array([], np.uint8),
                 normals=np.array([], np.float32)):
        self.points = points
        self.colors = colors
        self.normals = normals

    def torch(self):
        return PointCloud(torch.from_numpy(self.points),
                          torch.from_numpy(self.colors),
                          torch.from_numpy(self.normals))

    def copy(self):
        return PointCloud(self.points.copy(),
                          self.colors.copy(),
                          self.normals.copy())

    def transform(self, matrix):
        if self.points.size == 0:
            return

        self.points = Homogeneous(matrix) @ self.points
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T

        if self.normals.size > 0:
            self.normals = (
                normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    def to_open3d(self, compute_normals=False):
        """Converts the point to cloud to a :obj:`open3d.PointCloud`.

        Args:

            compute_normals (bool, optional): If `True`, use open3d
             to calculate the normals. Default is `False`.

        Returns:
            (:obj:`open3d.PointCloud`): Open3D point cloud.
        """
        from open3d import estimate_normals, KDTreeSearchParamHybrid, Vector3dVector
        from open3d import PointCloud as o3dPointCloud

        pcl = o3dPointCloud()

        if self.is_empty():
            return pcl

        pcl.points = Vector3dVector(self.points.squeeze())
        pcl.colors = Vector3dVector(self.colors.squeeze())

        if self.normals.size > 0:
            pcl.normals = Vector3dVector(self.normals)
        elif compute_normals:
            estimate_normals(pcl, search_param=KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            self.normals = np.array(pcl.normals, dtype=np.float32)

        return pcl

    def is_empty(self):
        """Returns whatever the point cloud is empty.
        """
        return self.points.size == 0

    @property
    def size(self):
        return self.points.shape[0]


def pcl_stack(pcl_list):
    pcl_list = [pcl for pcl in pcl_list
                if not pcl.is_empty()]
    if not pcl_list:
        return PointCloud()

    point_count = sum((pcl.points.shape[0] for pcl in pcl_list))
    normal_count = sum((pcl.normals.shape[0] for pcl in pcl_list))
    color_count = sum((pcl.colors.shape[0] for pcl in pcl_list))

    if normal_count > 0 and normal_count != point_count:
        raise RuntimeError(
            """Point and normal size are different, maybe you
            have point clouds with and without normals""")

    if color_count > 0 and color_count != point_count:
        raise RuntimeError(
            """Point and color counts are different, maybe you
            have point clouds with and without colors""")

    points = np.vstack([pcl.points for pcl in pcl_list
                        if not pcl.is_empty()])
    normals = np.vstack([pcl.normals for pcl in pcl_list
                         if not pcl.is_empty()])
    colors = np.vstack([pcl.colors for pcl in pcl_list
                        if not pcl.is_empty()])
    return PointCloud(points, colors, normals)


def from_open3d(pcl):
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

    from open3d import estimate_normals, KDTreeSearchParamHybrid, Vector3dVector
    from open3d import PointCloud as o3dPointCloud

    pcl = o3dPointCloud()

    pcl.points = Vector3dVector(points.squeeze())

    if colors is not None:
        pcl.colors = Vector3dVector(colors)

    if normals is not None:
        pcl.normals = Vector3dVector(normals)

    return pcl
