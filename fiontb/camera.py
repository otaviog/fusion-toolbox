"""Intrinsic and extrinsic camera handling.
"""

import numpy as np


class KCamera:
    """Intrinsic camera information.

    Attributes:

        matrix (:obj:`np.ndarray`): A 3x3 intrinsic camera
         transformation. Converts from image's column and row
         (0..img.width and 0..img.height) to u and v in camera space.

        undist_coeff (List[float], optional): Radial distortion
         coeficients. Default is `[]`.

        depth_radial_distortion (bool, optional): Correct radial
         distorion on Z values from datasets like the
         UCL-NUIM. Default is `False`.

        image_size ((int, int), optional): Width and height of the
         produced image. Default is `None`.
    """

    def __init__(self, matrix, undist_coeff=None, depth_radial_distortion=False, image_size=None):
        self.matrix = matrix

        if undist_coeff is not None:
            self.undist_coeff = undist_coeff
        else:
            self.undist_coeff = []

        self.depth_radial_distortion = depth_radial_distortion
        self.image_size = image_size

    @classmethod
    def create_from_params(cls, flen_x, flen_y, center_point,
                           undist_coeff=None, depth_radial_distortion=False, image_size=False):
        """Computes the intrinsic matrix from given focal lengths and center point information.

        Args:

            flen_x (float): X-axis focal length.

            flen_y (float): Y-axis focal length.

            center_point (float, float): Camera's central point on
            image space.

            undist_coeff (List[float], optional): Radial distortion
             coeficients. Default is `[]`.

            depth_radial_distortion (bool, optional): Correct radial
             distorion on Z values from datasets like the
             UCL-NUIM. Default is `False`.

            image_size ((int, int), optional): Width and height of the
             produced image. Default is `None`.

        """
        center_x, center_y = center_point
        k_trans = np.array([[1.0, 0.0, center_x],
                            [0.0, 1.0, center_y],
                            [0.0, 0.0, 1.0]])
        k_scale = np.array([[flen_x, 0.0, 0.0],
                            [0.0, flen_y, 0.0],
                            [0.0, 0.0, 1.0]])
        return cls(np.matmul(k_trans, k_scale), undist_coeff,
                   depth_radial_distortion, image_size)

    def unproject_image_to_cam(self, points):
        xyz_coords = points[:, 0:2]
        xyz_coords = np.insert(xyz_coords, 2, 1.0, axis=1)
        xyz_coords = np.matmul(np.linalg.inv(
            self.matrix), xyz_coords)

        depths = points[:, 2, 0]

        if self.depth_radial_distortion:
            depths = (depths /
                      np.sqrt(np.power(xyz_coords[:, 0:2, 0], 2).sum(1) + 1))

        xyz_coords[:, 0:2, 0] *= np.vstack([depths, depths]).T
        xyz_coords[:, 2, 0] = depths

        return xyz_coords

    def project_cam_to_image(self, points):
        points = np.matmul(self.matrix, points)

        z = points[:, 2, 0]
        z = np.vstack([z, z]).T
        points[:, 0:2, 0] /= z
        return points

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class RTCamera:
    """Extrinsic camera transformation.

    Wrappers the Rotation and Transalation transformation of a camera.

    Attributes:

        matrix (np.array): A 4x4 matrix representing the camera space to world
         space transformation.

    """

    def __init__(self, matrix):
        self.matrix = matrix

    @classmethod
    def create_from_params(cls, position, rotation_matrix):
        """Constructor using position vector and rotation matrix.

        Args:

            Those vectors should convert from camera space to world space.
        """
        posx, posy, posz = position
        g_trans = np.array([[1.0, 0.0, 0.0, posx],
                            [0.0, 1.0, 0.0, posy],
                            [0.0, 0.0, 1.0, posz],
                            [0.0, 0.0, 0.0, 1.0]])
        g_rot = np.eye(4, 4)
        g_rot[0:3, 0:3] = rotation_matrix
        return cls(np.matmul(g_trans, g_rot))

    def transform_cam_to_world(self, points):
        """Transform points from camera to world space.

        Args:

            points (:obj:`numpy.ndarray`): Array of shape [N, 3], [N,
             3, 1] or [3] with 1 or more points in world space.

        Returns:

            (:obj:`numpy.ndarray`): Transformed points into world
             space. Shape is the same as input.

        """

        points = np.insert(points, 3, 1, axis=1)
        points = np.matmul(self.matrix, points)
        points = np.delete(points, 3, 1)
        return points

    def transform_world_to_cam(self, points):
        """Transform points from world to camera space.

        Args:

            points (:obj:`numpy.ndarray`): Array of shape [N, 3], [N,
             3, 1] or [3] with 1 or more points in world space.

        Returns:

            (:obj:`numpy.ndarray`): Transformed points into camera
             space. Shape is the same as input.

        """
        waxis = 1
        if len(points.shape) == 1:
            waxis = 0
        elif len(points.shape) == 2:
            points = points[..., np.newaxis]

        points = np.insert(points, 3, 1, axis=waxis)
        points = np.matmul(np.linalg.inv(self.matrix), points)
        points = np.delete(points, 3, waxis)
        return points

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
