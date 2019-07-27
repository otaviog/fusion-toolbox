"""Lie algebra operators for SO3 group.
"""

import math

import torch

# pylint: disable=invalid-name


def hat(vector3):
    """Hat operator: return a skew matrix representing the cross product
    with a given vector.

    """
    return torch.tensor([[0.0, -vector3[2], vector3[1]],
                         [vector3[2], 0.0, -vector3[0]],
                         [-vector3[1], vector3[0], 0.0]],
                        dtype=vector3.dtype)


def vee(mat3):
    return torch.tensor([mat3[1, 2],
                         mat3[0, 2],
                         mat3[1, 0]], dtype=mat3.dtype)


_I3 = torch.eye(3)


def exp(twist):
    theta = twist.norm()
    omega = hat(twist)
    so3 = (_I3
           + (math.sin(theta) / theta) * omega
           + (1.0 - math.cos(theta)) / (theta*theta) * omega @ omega)
    return so3
