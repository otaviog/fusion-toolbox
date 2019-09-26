"""Intrinsic and extrinsic camera handling.
"""
import math

import numpy as np
import quaternion
import torch
import tenviz

from fiontb._utils import ensure_torch, empty_ensured_size
from fiontb._cfiontb import (project_op_forward as _project_op_forward,
                             project_op_backward as _project_op_backward)

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

        image_size ((int, int), optional): Width and height of the
         produced image. Default is `None`.
    """

    def __init__(self, matrix, undist_coeff=None, image_size=None):
        self.matrix = ensure_torch(matrix).float()

        if undist_coeff is not None:
            self.undist_coeff = undist_coeff
        else:
            self.undist_coeff = []

        self.image_size = image_size

    @classmethod
    def from_json(cls, json):
        """Loads from json representaion.
        """
        return cls(torch.tensor(json['matrix'], dtype=torch.float).view(-1, 3),
                   undist_coeff=json.get('undist_coeff', None),
                   image_size=json.get('image_size', None))

    def to_json(self):
        """Converts the camera intrinsics to its json dict representation.

        Returns: (dict): Dict ready for json dump.

        """
        json = {
            'matrix': self.matrix.tolist(),
        }

        if self.undist_coeff is not None:
            json['undist_coeff'] = self.undist_coeff

        if self.image_size is not None:
            json['image_size'] = self.image_size

        return json

    @classmethod
    def from_params(cls, flen_x, flen_y, center_point,
                    undist_coeff=None, image_size=None):
        """Computes the intrinsic matrix from given focal lengths and center point information.

        Args:

            flen_x (float): X-axis focal length.

            flen_y (float): Y-axis focal length.

            center_point (float, float): Camera's central point on
            image space.

            undist_coeff (List[float], optional): Radial distortion
             coeficients. Default is `[]`.

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
        return cls(k_trans @ k_scale, undist_coeff, image_size)

    @classmethod
    def from_estimation_by_fov(cls, hfov, vfov, img_width, img_height):
        """Create intrinsics using the ratio of image pixel dimensions and
        field of view angles. This is useful for guessing intrinsics
        from a supplied field of views. However, without the actual
        sensor width and height, it's not possible to generate
        accurate focal lengths.

        Args:

            hfov (float): Horizontal field of view in radians.

            vfov (float): Vertical field of view in radians.

            img_width (int): Image width in pixels.

            img_height (int): Image height in pixels.
        """

        flen_x = img_width * .5 / math.tan(hfov/2)
        flen_y = img_height * .5 / math.tan(vfov/2)

        return cls.from_params(flen_x, flen_y, (img_width*.5, img_height*.5))

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

    def scaled(self, xscale, yscale=None):
        if yscale is None:
            yscale = xscale
        return KCamera.from_params(self.matrix[0, 0]*xscale, self.matrix[1, 1]*yscale,
                                   (self.matrix[0, 2]*xscale,
                                    self.matrix[1, 2]*yscale),
                                   self.undist_coeff, self.image_size).to(self.matrix.device)

    def to(self, device):
        return KCamera(self.matrix.to(device), self.undist_coeff,
                       self.image_size)

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

    def get_opengl_projection_matrix(self, near, far, dtype=torch.float):
        return torch.from_numpy(tenviz.projection_from_kcam(
            self.matrix, near, far).to_matrix()).to(dtype)


class Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, intrinsics):
        ctx.save_for_backward(points, intrinsics)
        return _project_op_forward(points, intrinsics)

    @staticmethod
    def backward(ctx, dy_grad):
        points, intrinsics = ctx.saved_tensors
        return _project_op_backward(dy_grad, points, intrinsics), None


class RigidTransform:
    """Helper class to multiply [Nx3] points by [4x4] matrices.
    """

    def __init__(self, matrix):
        self.matrix = matrix

    def __matmul__(self, points):
        points = self.matrix[:3, :3].matmul(points.view(-1, 3, 1))
        points += self.matrix[:3, 3].view(3, 1)

        return points.squeeze()


class IRigidTransform:
    def __init__(self):
        self.out_tensor = None
        self.matrix = None

    def __call__(self, matrix):
        self.matrix = matrix
        return self

    def __matmul__(self, points):
        points = points.view(-1, 3, 1)
        self.out_tensor = empty_ensured_size(self.out_tensor, points.size(0),
                                             3, 1,
                                             dtype=points.dtype,
                                             device=points.device)
        torch.matmul(self.matrix[:3, :3], points, out=self.out_tensor)
        self.out_tensor += self.matrix[:3, 3].view(3, 1)

        return self.out_tensor.squeeze()


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
        return cls(torch.tensor(json['matrix'], dtype=torch.float).view(-1, 4))

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

    def integrate(self, matrix):
        return RTCamera(matrix @ self.matrix)

    def translate(self, tx, ty, tz):
        return RTCamera(self.matrix @ torch.tensor([[1, 0, 0, tx],
                                                    [0, 1, 0, ty],
                                                    [0, 0, 1, tz],
                                                    [0, 0, 0, 1]]))

    def inverse(self):
        return RTCamera(self.world_to_cam)

    def clone(self):
        return RTCamera(self.matrix.clone())

    @property
    def center(self):
        return self.matrix[:3, 3]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __matmul__(self, other):
        return RTCamera(self.matrix @ other.matrix)
