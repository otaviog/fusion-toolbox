import numpy as np
import torch

from .camera import RigidTransform, normal_transform_matrix


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
                   torch.from_numpy((np.array(pcl.colors)*255.0)).byte(),
                   torch.from_numpy(np.array(pcl.normals, dtype=np.float32)))

    @staticmethod
    def from_frame(frame, world_space=True, compute_normals=True):
        from .frame import FramePointCloud

        fpcl = FramePointCloud.from_frame(frame)
        return (fpcl.unordered_point_cloud(
            world_space=world_space,
            compute_normals=compute_normals), fpcl.mask)

    def __init__(self, points, colors=None, normals=None):
        self.points = points
        self.colors = colors
        self.normals = normals

    def clone(self):
        return PointCloud(
            self.points.clone(),
            self.colors.clone() if self.colors is not None else None,
            self.normals.clone() if self.normals is not None else None)

    def transform(self, matrix):
        if self.points.size == 0:
            return PointCloud(self.points)

        rit = RigidTransform(matrix)
        points = rit @ self.points

        normals = None
        if self.normals is not None:
            normals = rit.transform_normals(self.normals)

        return PointCloud(points, self.colors, normals)

    def itransform(self, matrix):
        transform = RigidTransform(matrix.float().to(self.device))
        transform.inplace(self.points)
        if self.normals is not None:
            transform.inplace_normals(self.normals)

    def index_select(self, index):
        return PointCloud(
            self.points[index],
            self.colors[index] if self.colors is not None else None,
            self.normals[index] if self.normals is not None else None)

    def to_open3d(self, compute_normals=False):
        """Converts the point to cloud to a :obj:`open3d.PointCloud`.

        Args:

            compute_normals (bool, optional): If `True`, use open3d
             to calculate the normals. Default is `False`.

        Returns:
            (:obj:`open3d.PointCloud`): Open3D point cloud.
        """
        import open3d as o3d

        pcl = o3d.geometry.PointCloud()

        if self.is_empty():
            return pcl

        pcl.points = o3d.utility.Vector3dVector(self.points.numpy())
        pcl.colors = o3d.utility.Vector3dVector(
            self.colors.float().numpy()/255.0)

        if self.normals is not None:
            pcl.normals = o3d.utility.Vector3dVector(self.normals.numpy())
        elif compute_normals:
            estimate_normals(pcl, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            self.normals = torch.from_numpy(
                np.array(pcl.normals, dtype=np.float32))

        return pcl

    def is_empty(self):
        """Returns whatever the point cloud is empty.
        """
        return self.points.size == 0

    def to(self, dst):
        return PointCloud(self.points.to(dst),
                          (self.colors.to(dst) if self.colors is not None else None),
                          (self.normals.to(dst) if self.normals is not None else None))

    @property
    def size(self):
        return self.points.shape[0]

    @property
    def device(self):
        return self.points.device

    def __getitem__(self, *args):
        return PointCloud(
            self.points[args],
            self.colors[args] if self.colors is not None else None,
            self.normals[args] if self.normals is not None else None)


def stack_pcl(pcl_list):
    pcl_list = [pcl for pcl in pcl_list
                if not pcl.is_empty()]
    if not pcl_list:
        return PointCloud(torch.Tensor([], dtype=torch.float))

    point_count = sum((pcl.points.size(0) for pcl in pcl_list))
    normal_count = sum((pcl.normals.size(0) for pcl in pcl_list
                        if pcl.normals is not None))
    color_count = sum((pcl.colors.size(0) for pcl in pcl_list
                       if pcl.colors is not None))

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
