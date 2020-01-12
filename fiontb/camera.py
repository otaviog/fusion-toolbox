"""Intrinsic and extrinsic camera handling.
"""
import math
import copy

import numpy as np
import quaternion
import torch
import tenviz

from fiontb._utils import ensure_torch, empty_ensured_size
from fiontb._cfiontb import (ProjectOp as _ProjectOp,
                             RigidTransformOp as _RigidTransformOp)

_GL_HAND_MTX = torch.eye(4, dtype=torch.float)
_GL_HAND_MTX[2, 2] = -1


class KCamera:
    """Intrinsic pinhole camera model for projecting and backprojecting points.

    Attributes:

        matrix (:obj:`torch.Tensor`): A 3x3 intrinsic camera
         transformation. Converts from image's column and row
         (0..img.width and 0..img.height) to u and v in camera space.

        undist_coeff (List[float], optional): Radial distortion
         coefficients. Default is `[]`.

        image_size ((int, int), optional): Width and height of the
         produced image. Default is `None`.

    """

    def __init__(self, matrix, undist_coeff=None, image_size=None):
        self.matrix = ensure_torch(matrix)

        if undist_coeff is not None:
            self.undist_coeff = undist_coeff
        else:
            self.undist_coeff = []

        self.image_size = image_size

    @classmethod
    def from_json(cls, json):
        r"""Loads from the FTB JSON representation.

        """
        return cls(torch.tensor(json['matrix'], dtype=torch.float).view(-1, 3),
                   undist_coeff=json.get('undist_coeff', None),
                   image_size=json.get('image_size', None))

    def to_json(self):
        r"""Converts the camera intrinsics to its FTB JSON dict representation.

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
        """Computes the intrinsic matrix from given focal lengths and center
        point information.

        Args:

            flen_x (float): X-axis focal length.

            flen_y (float): Y-axis focal length.

            center_point (float, float): Camera's central point on
            image space.

            undist_coeff (List[float], optional): Radial distortion
             coefficients. Default is `[]`.

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
        """Back project 2D image coordinates and its z value into 3D camera space.

        Args:

            points (:obj:`torch.Tensor`): Array (N x 3) of 2D image points and z values.
             The columns are expected to represent u, v and z.

        Returns:
            (:obj:`torch::Tensor`): Array (N x 3) of 3D points in camera space.
        """

        xyz_coords = points.clone()
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]
        cx = self.matrix[0, 2]
        cy = self.matrix[1, 2]

        z = xyz_coords[:, 2]
        xyz_coords[:, 0] = (xyz_coords[:, 0] - cx) * z / fx
        xyz_coords[:, 1] = (xyz_coords[:, 1] - cy) * z / fy

        return xyz_coords[:, :2]

    def project(self, points):
        """Project 3D points in camera space to image space.

        Applies division by z.

        Args:

            points (:obj:`torch.Tensor`): Array (Nx3) of 3D points in camera space.

        Returns:
            (:obj:`torch.Tensor`): Array (Nx2) of 2D image points.

        """

        points = (self.matrix @ points.reshape(-1, 3, 1)).reshape(-1, 3)

        z = points[:, 2]

        points[:, :2] /= z.reshape(-1, 1)
        return points

    def scaled(self, xscale, yscale=None):
        """
        Returns intrinsic parameters adjusted for a new size scale.

        Args:
            xscale (float):  Horizontal scaling factor. Global scale if yscale is not specified.
            yscale (float, optional): Vertical scaling factor, if not specified, then the same scale for x is used.

        Returns:
            (:obj:`KCamera`): Scaled intrinsic parameters.
        """
        if yscale is None:
            yscale = xscale

        image_size = None
        if self.image_size is not None:
            image_size = (int(self.image_size[0]*xscale),
                          int(self.image_size[1]*yscale))

        return KCamera.from_params(
            self.matrix[0, 0]*xscale, self.matrix[1, 1]*yscale,
            (self.matrix[0, 2]*xscale, self.matrix[1, 2]*yscale),
            self.undist_coeff,
            image_size).to(self.matrix.device)

    def clone(self):
        """Create a copy of this instance.

        Returns: (:obj:`KCamera`): Copy.
        """
        return KCamera(self.matrix.clone(),
                       copy.deepcopy(self.undist_coeff),
                       self.image_size)

    def get_projection_params(self, near, far):

        return tenviz.Projection.from_intrinsics(
            self.matrix, near, far)

    def get_opengl_projection_matrix(self, near, far, dtype=torch.float):
        """Converts this camera intrinsic to its OpenGL matrix version.

        Args:

            near (float): Near clipling plane distance.

            far (float): Far cliping plane distance.

            dtype (:obj:`torch.dtype`, optional): Specifies the returned matrix dtype.

        Returns:
            (:obj:`torch.Tensor`): A (4 x 4) OpenGL projection matrix.
        """
        return torch.from_numpy(
            self.get_projection_params(near, far).to_matrix()).to(dtype)

    def to(self, dst):
        return KCamera(self.matrix.to(dst), self.undist_coeff,
                       self.image_size)

    @property
    def image_width(self):
        """

        Returns:
            (int) image width in pixels.

        """
        return self.image_size[0]

    @property
    def image_height(self):
        """

        Returns:
            (int) image height in pixels.

        """
        return self.image_size[1]

    @property
    def device(self):
        """

        Returns:
            (str): matrix's torch device

        """
        return self.matrix.device

    @property
    def pixel_center(self):
        """Center pixel. 

        Returns: (float, float): X and Y coordinates.
        """
        return (self.matrix[0, 2].item(), self.matrix[1, 2].item())

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        if not isinstance(other, KCamera):
            return False

        return (torch.all(self.matrix == other.matrix)
                and (self.undist_coeff == other.undist_coeff)
                and (self.image_size == other.image_size))


class Project(torch.autograd.Function):
    """
    Differentiable projection operator for pinhole camera model
    """
    @staticmethod
    def forward(ctx, points, intrinsics):
        """

        Args:
            ctx: torch's context
            points (:obj:`torch.Tensor`): Array (N x 3) of 3D points in camera space
            intrinsics (:obj:`torch.Tensor`): Matrix (3 x 3) of camera intrinsics.

        Returns:
            (:obj:`torch.Tensor`): Array (N x 2) of 2D points in image space.

        """
        ctx.save_for_backward(points, intrinsics)
        return _ProjectOp.forward(points, intrinsics)

    @staticmethod
    def backward(ctx, dy_grad):
        """
        Backward implementation.

        Args:
            dy_grad:
        """
        points, intrinsics = ctx.saved_tensors
        return _ProjectOp.backward(dy_grad, points, intrinsics), None


class RigidTransform:
    """Helper object for multiplying (4 x 4) or (3 x 4) matrices and (N x
    3) points. Use its matmul operator for applying the transform.

    Example:

        >>> result = RigidTransform(torch.rand(4, 4)) @ torch.rand(100, 3)
        >>> result.size(0), result.size(1)
        100, 3

    Attributes:

        matrix (torch.Tensor): The left side matrix (4 x 4) or (3 x 4).

    """

    def __init__(self, matrix):
        self.matrix = matrix
        self._normal_matrix = None

    def __matmul__(self, points):
        points = self.matrix[:3, :3].matmul(points.view(-1, 3, 1))
        points += self.matrix[:3, 3].view(3, 1)

        return points.squeeze()

    def transform_normals(self, normals):
        """Transform a normal vector array by the instance's matrix.

        Args:

            normals (torch.Tensor): A (Nx3) normal vectors array.

        Returns: (torch.Tensor): Transformed normal vectors (Nx3).


        """
        return (self.normal_matrix @ normals.view(-1, 3, 1)).view(-1, 3)

    def inplace(self, points):
        """Transform the points in-place.

        Args:
        
            points (torch.Tensor): Input points array (Nx3).
        
        """
        points = points.view(-1, 3)
        _RigidTransformOp.transform_inplace(self.matrix, points)
        return points

    @property
    def normal_matrix(self):
        """The transpose of the inverse of the input matrix for
        correct transforming normal vectors.

        Returns: (torch.Tensor): A (3 x 3) matrix.

        """
        if self._normal_matrix is None:
            self._normal_matrix = normal_transform_matrix(self.matrix)

        return self._normal_matrix

    def inplace_normals(self, normals):
        """Transform the normals in-place.
        
        Args: 
        
            normals (torch.Tensor): Input normals array (Nx3).

        """
        normals = normals.view(-1, 3)
        _RigidTransformOp.transform_normals_inplace(self.matrix, normals)
        return normals

    def rodrigues(self):
        """Computes the Rodrigues' rotation representation of the matrix.

        Returns: (torch.Tensor): A 3 sized tensor containing the
         Rodrigues' rotational representation.

        """
        rodrigues = torch.empty(3, dtype=self.matrix.dtype)
        _RigidTransformOp.rodrigues(self.matrix.cpu(), rodrigues)        
        return rodrigues

    def translation(self):
        """Translation matrix part.
        
        Returns: (torch.Tensor): A 3 sized tensor containing the X, Y
        and Z translation.

        """
        return self.matrix[:3, 3]


def normal_transform_matrix(matrix):
    r"""Returns the transpose of the inverse of the given matrix. The
    resulting matrix will preserve normal vector orientation and size.

    Args:

        matrix: A (4 x 4) or (3 x 4) affine transformation matrix.

    Returns: (:obj:`torch.Tensor`): Rotation only (3 x 3) matrix for .

    """
    return torch.inverse(matrix[:3, :3]).transpose(1, 0)


class RTCamera:
    """Extrinsic camera wrapper.

    Attributes:

        matrix (:obj:`torch.Tensor`): A (4 x 4) matrix that transforms from camera space into world
         space. Type is double precision float.

    """

    def __init__(self, matrix=None):
        if matrix is not None:
            self.matrix = ensure_torch(matrix, dtype=torch.double)
        else:
            self.matrix = torch.eye(4, dtype=torch.double)

    @classmethod
    def create_from_pos_rot(cls, position, rotation_matrix):
        """Construct from a position vector and a rotation matrix.

        Args:

            position ((float, float, float)): Translation part.
            rotation_matrix (:obj:`torch.Tensor`): Rotation matrix (3 x 3).
        """
        posx, posy, posz = position
        g_trans = torch.tensor([[1.0, 0.0, 0.0, posx],
                                [0.0, 1.0, 0.0, posy],
                                [0.0, 0.0, 1.0, posz],
                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.double)
        g_rot = torch.eye(4, dtype=torch.double)
        g_rot[0:3, 0:3] = rotation_matrix.double()
        return cls(g_trans @ g_rot)

    @classmethod
    def create_from_pos_quat(cls, x, y, z, qw, qx, qy, qz):
        """
        Constructs from position and quaternion.

        """
        rot_mtx = torch.from_numpy(quaternion.as_rotation_matrix(
            np.quaternion(qw, qx, qy, qz))).double()

        cam_mtx = np.eye(4, dtype=torch.double)
        cam_mtx[0:3, 0:3] = rot_mtx
        cam_mtx[0:3, 3] = [x, y, z]

        return cls(cam_mtx)

    @classmethod
    def from_json(cls, json):
        """
        Constructs from FTB's JSON representation

        Args:
            json (dict): JSON dictionary.

        Returns:

        """
        return cls(torch.tensor(json['matrix'], dtype=torch.double).view(-1, 4))

    def to_json(self):
        """
        Converts to FTB's JSON representation. See `fiontb.data.load_ftb`.


        Returns:
            (dict): JSON dict.

        """
        return {'matrix': self.matrix.tolist()}

    @property
    def cam_to_world(self):
        """Matrix with camera to world transformation

        Returns:
            (:obj:`torch.Tensor`): A (4 x 4) float64 matrix.
        """
        return self.matrix

    @property
    def world_to_cam(self):
        """Matrix with world to camera transformation.

        Returns:
            (:obj:`torch.Tensor`): A (4 x 4) float64 matrix.
        """
        return torch.inverse(self.matrix)

    @property
    def opengl_view_cam(self):
        """
        Get a OpenGL ready view matrix like this instance.

        Returns:
            (:obj:`torch.Tensor`): A (4 x 4) float32 matrix.

        """
        return (_GL_HAND_MTX @ self.world_to_cam.float()).float()

    def right_transform(self, matrix):
        """
        Multiply with this order: matrix @ self.matrix

        Args:
            matrix (:obj:`torch.Tensor`): Transformation matrix (4 x 4).

        Returns:
            (:obj:`RTCamera`): New transformed camera.
        """
        return RTCamera(matrix.cpu().double() @ self.matrix)

    def transform(self, matrix):
        """
        Transforms the camera by other matrix.

        Args:
            matrix (:obj:`torch.Tensor`): A (4x4) matrix.

        Returns:
            (:obj:`RTCamera`): New transformed camera.
        """
        return RTCamera(self.matrix @ matrix.cpu().double())

    def __matmul__(self, other):
        return RTCamera(self.matrix @ other.matrix.double())

    def translate(self, tx, ty, tz):
        """Translate the camera position.

        Args:
            tx (float): X translation.
            ty (float): Y translation.
            tz (float): Z translation.

        Returns:
            (:obj:`RTCamera`): Translated camera.

        """
        return RTCamera(self.matrix @ torch.tensor([[1, 0, 0, tx],
                                                    [0, 1, 0, ty],
                                                    [0, 0, 1, tz],
                                                    [0, 0, 0, 1]], dtype=torch.double))
    def difference(self, other):
        """
        Computes the he relative transformation between this camera to other.


        Args:
            other (:obj:`RTCamera`): The target camera pose.

        Returns:
            (:obj:`RigidTransform`): Relative transformation.

        """
        return RigidTransform(self.world_to_cam * other.matrix)

    @property
    def center(self):
        """

        Returns:
            ((float, float, float)): X, Y and Z camera position.
        """
        return (self.matrix[0, 3].item(), self.matrix[0, 3].item(), self.matrix[0, 3].item())

    def clone(self):
        """
        Clone the instance.

        Returns:
            (:obj:`RTCamera`): Copy.

        """
        return RTCamera(self.matrix.clone())

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
