"""Intrinsic and extrinsic camera handling.
"""

import numpy as np
import quaternion
import torch

from fiontb._utils import ensure_torch

_GL_HAND_MTX = torch.eye(4, dtype=torch.float)
_GL_HAND_MTX[2, 2] = -1


class KCamera:
    """Intrinsic pinhole camera model.

    Attributes:

        matrix (:obj:`torch.Tensor`): A 3x3 intrinsic camera
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
        self.matrix = ensure_torch(matrix).float()

        if undist_coeff is not None:
            self.undist_coeff = undist_coeff
        else:
            self.undist_coeff = []

        self.depth_radial_distortion = depth_radial_distortion
        self.image_size = image_size

    @classmethod
    def from_json(cls, json):
        """Loads from json representaion.
        """
        return cls(torch.tensor(json['matrix'], dtype=torch.float),
                   undist_coeff=json.get('undist_coeff', None),
                   depth_radial_distortion=json['is_radial_depth'],
                   image_size=json.get('image_size', None))

    def to_json(self):
        """Converts the camera intrinsics to its json dict representation.

        Returns: (dict): Dict ready for json dump.

        """
        json = {
            'matrix': self.matrix.tolist(),
            'is_radial_depth': self.depth_radial_distortion
        }

        if self.undist_coeff is not None:
            json['undist_coeff'] = self.undist_coeff

        if self.image_size is not None:
            json['image_size'] = self.image_size

        return json

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
        k_trans = torch.tensor([[1.0, 0.0, center_x],
                                [0.0, 1.0, center_y],
                                [0.0, 0.0, 1.0]])
        k_scale = torch.tensor([[flen_x, 0.0, 0.0],
                                [0.0, flen_y, 0.0],
                                [0.0, 0.0, 1.0]])
        return cls(k_trans @ k_scale, undist_coeff,
                   depth_radial_distortion, image_size)

    def backproject(self, points):
        """Project image points to the camera space.
        """

        xyz_coords = points.clone()
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]
        cx = self.matrix[0, 2]
        cy = self.matrix[1, 2]

        z = xyz_coords[:, 2]
        xyz_coords[:, 0] = (xyz_coords[:, 0] - cx) * z / fx
        xyz_coords[:, 1] = (xyz_coords[:, 1] - cy) * z / fy

        return xyz_coords

    def project(self, points):
        """Project camera to image space.
        """

        points = (self.matrix @ points.reshape(-1, 3, 1)).reshape(-1, 3)

        z = points[:, 2]

        points[:, :2] /= z.reshape(-1, 1)
        return points

    @property
    def pixel_center(self):
        """Center pixel.
        """
        return (self.matrix[0, 2], self.matrix[1, 2])

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, KCamera):
            return False

        return (torch.all(self.matrix == other.matrix)
                and (self.undist_coeff == other.undist_coeff)
                and (self.depth_radial_distortion == other.depth_radial_distortion)
                and (self.image_size == other.image_size))


class Homogeneous:
    """Helper class to multiply [Nx3] or [Nx3x1] points by a [4x4] matrix.
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def __matmul__(self, points):
        points = self.matrix[:3, :3] @ points.reshape(-1, 3, 1)
        points += self.matrix[:3, 3].reshape(3, 1)

        return points.squeeze()


def normal_transform_matrix(matrix):
    """Returns the transposed inverse of transformation matrix. Suitable
    for transforming normals.

    Args:

        matrix: [4x4] affine transformation matrix.

    Returns: (:obj:`torch.Tensor`): Rotation only [3x3] matrix.

    """
    return torch.inverse(matrix[:3, :3]).transpose(1, 0)


class RTCamera:
    """Extrinsic camera transformation.

    Wrappers the Rotation and Transalation transformation of a camera.

    Attributes:

        matrix (:obj:`torch.Tensor`): A 4x4 matrix representing the camera space to world
         space transformation.

    """

    def __init__(self, matrix):
        self.matrix = ensure_torch(matrix).float()

    @classmethod
    def create_from_pos_rot(cls, position, rotation_matrix):
        """Constructor using position vector and rotation matrix.

        Args:

            Those vectors should convert from camera space to world space.
        """
        posx, posy, posz = position
        g_trans = torch.tensor([[1.0, 0.0, 0.0, posx],
                                [0.0, 1.0, 0.0, posy],
                                [0.0, 0.0, 1.0, posz],
                                [0.0, 0.0, 0.0, 1.0]])
        g_rot = torch.eye(4)
        g_rot[0:3, 0:3] = rotation_matrix
        return cls(g_trans @ g_rot)

    @classmethod
    def create_from_pos_quat(cls, x, y, z, qw, qx, qy, qz):
        # TODO
        g_trans = torch.tensor([[1.0, 0.0, 0.0, x],
                                [0.0, 1.0, 0.0, y],
                                [0.0, 0.0, 1.0, z],
                                [0.0, 0.0, 0.0, 1.0]])
        g_rot = torch.eye(4)
        g_rot[0:3, 0:3] = torch.from_numpy(quaternion.as_rotation_matrix(
            np.quaternion(qw, qx, qy, qz)))

        # return cls(g_trans @ g_rot)

        rot_mtx = torch.from_numpy(quaternion.as_rotation_matrix(
            np.quaternion(qw, qx, qy, qz)))

        cam_mtx = np.eye(4)
        cam_mtx[0:3, 0:3] = rot_mtx
        cam_mtx[0:3, 3] = [x, y, z]

        return cls(cam_mtx)

    @classmethod
    def from_json(cls, json):
        return cls(torch.tensor(json['matrix'], dtype=torch.float))

    def to_json(self):
        return {'matrix': self.matrix.tolist()}

    @property
    def cam_to_world(self):
        """Matrix with camera to world transformation
        """
        return self.matrix

    @property
    def world_to_cam(self):
        """Matrix with world to camera transformation
        """
        return torch.inverse(self.matrix)

    @property
    def opengl_view_cam(self):
        return _GL_HAND_MTX @ self.world_to_cam

    def integrate(self, rt_cam):
        self.matrix = rt_cam.matrix @ self.matrix

    def translate(self, tx, ty, tz):
        return RTCamera(self.matrix @ torch.tensor([[1, 0, 0, tx],
                                                    [0, 1, 0, ty],
                                                    [0, 0, 1, tz],
                                                    [0, 0, 0, 1]]))

    @property
    def center(self):
        return self.matrix[:3, 3]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
