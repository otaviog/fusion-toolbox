"""Point cloud structure.
"""

import torch
import numpy as np

from .camera import RigidTransform, normal_transform_matrix


class PointCloud:
    """Point cloud data structure.

    Point, color and normals are stored as torch tensor. Slicing
     operator is avaliable for capturing a new point cloud.

    Attributes:

        points (:obj:`torch.Tensor`): 3D points (N x 3) of float.

        colors (:obj:`torch.Tensor`): Point colors (N x 3) of uint8.

        normals (:obj:`torch.Tensor`): Point normal vectors (N x 3) of float.

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
        """Creates a point cloud from a frame.

        Args:

            world_space (bool, optional): `True` to set the points on
             world space, according to frame's extrinsic
             parameters. Default is `True`.

            compute_normals (bool, optional): Use normals. Default is `True`.

        Returns: ((PointCloud, torch.Tensor)): Point cloud and a (H x
         W) bool tensor mask of the frame pixels included in the point
         cloud.

        """
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
        """Transform the points and normals and return and new point cloud.

        Args:

            matrix (torch.Tensor): A (4 x 4) or (3 x 4) transformation matrix.

        Returns: (PointCloud): New transformed point cloud.
        """

        if self.points.size == 0:
            return PointCloud(self.points)

        rit = RigidTransform(matrix)
        points = rit @ self.points

        normals = None
        if self.normals is not None:
            normals = rit.transform_normals(self.normals)

        return PointCloud(points, self.colors, normals)

    def itransform(self, matrix):
        """Transform the points and normals inplace.

        Args:

            matrix (torch.Tensor): A (4 x 4) or (3 x 4) transformation matrix.
        """

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
        r"""Change the point cloud device or dtype. Dtype are applied only to
        points and normals.

        Args:

            dst (torch.dtype, torch.device, str): Dtype or torch device.

        Returns: (PointCloud): converted point cloud.
        """
        return PointCloud(self.points.to(dst),
                          (self.colors.to(dst) if self.colors is not None else None),
                          (self.normals.to(dst) if self.normals is not None else None))

    @property
    def size(self):
        r"""Returns the point cloud size

        """
        return self.points.shape[0]

    @property
    def device(self):
        r"""Returns the point cloud torch device.

        """
        return self.points.device

    def __getitem__(self, *args):
        """Slicing operator for all instance properties.

        Returns: (PointCloud): Sliced point cloud.
        """
        return PointCloud(
            self.points[args],
            self.colors[args] if self.colors is not None else None,
            self.normals[args] if self.normals is not None else None)

    def __len__(self):
        return self.size


def stack_pcl(pcl_list):
    """Concatenate point clouds into one.

    All point cloud need to have the same set of attributes, that is,
    it will raise a error when join point cloud that have normals with
    ones that don't.

    Args:

        pcl_list (List[PointCloud]): Point cloud list.

    Returns: (PointCloud): Joined point cloud.

    """

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
