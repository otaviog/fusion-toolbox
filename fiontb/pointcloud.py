import numpy as np
import torch

from .camera import Homogeneous, normal_transform_matrix


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
        return cls(torch.from_numpy(np.array(pcl.points, dtype=np.float32)),
                   torch.from_numpy(np.array(pcl.colors, dtype=np.uint8)),
                   torch.from_numpy(np.array(pcl.normals, dtype=np.float32)))

    def __init__(self, points,
                 colors=torch.tensor([], dtype=torch.uint8),
                 normals=torch.tensor([], dtype=torch.float32)):
        self.points = points
        self.colors = colors
        self.normals = normals

    def copy(self):
        return PointCloud(self.points.clone(),
                          self.colors.clone(),
                          self.normals.clone())

    def transform(self, matrix):
        if self.points.size == 0:
            return

        self.points = Homogeneous(matrix) @ self.points
        normal_matrix = normal_transform_matrix(matrix)

        if self.normals.numel() > 0:
            self.normals = (
                normal_matrix @ self.normals.reshape(-1, 3, 1)).squeeze()

    def index_select(self, index):
        return PointCloud(
            self.points[index],
            self.colors[index],
            self.normals[index])

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

        if self.normals.numel() > 0:
            pcl.normals = Vector3dVector(self.normals)
        elif compute_normals:
            estimate_normals(pcl, search_param=KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            self.normals = torch.from_numpy(
                np.array(pcl.normals, dtype=np.float32))

        return pcl

    def is_empty(self):
        """Returns whatever the point cloud is empty.
        """
        return self.points.size == 0

    @property
    def size(self):
        return self.points.shape[0]


def stack_pcl(pcl_list):
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

    points = torch.cat([pcl.points for pcl in pcl_list
                        if not pcl.is_empty()])
    normals = torch.cat([pcl.normals for pcl in pcl_list
                         if not pcl.is_empty()])
    colors = torch.cat([pcl.colors for pcl in pcl_list
                        if not pcl.is_empty()])
    return PointCloud(points, colors, normals)
