"""Unit testing of SO3 operations
"""
import unittest

import torch

import fiontb.registration.se3 as se3


class TestSE3(unittest.TestCase):
    """Test SE3 module.
    """
    # pylint: disable=no-self-use, redefined-builtin

    def _test_exp_rt_to_matrix(self):
        """Gradcheck the ExpRtToMatrix operator
        """
        torch.manual_seed(10)
        for dev in ["cpu:0",
                    #"cuda:0"
        ]:
            input = (torch.rand(1, 6, dtype=torch.double,
                                requires_grad=True, device=dev),)
            torch.autograd.gradcheck(
                se3.ExpRtToMatrix.apply,
                input, eps=1e-6, atol=1e-4,
                raise_exception=True)

    def test_exp_rt_transform(self):
        """Gradcheck the ExpRtTransform layer.
        """
        transform = se3.ExpRtTransform.apply
        torch.manual_seed(10)
        import math

        angle = math.pi / 3

        axis = torch.Tensor([0.0, 1.0, 0.0])
        axis /= axis.norm(2)
        axis *= angle

        translate = [14, 4.2, 2]
        translate = [0, 0, 0]
        #axis = [0, 0, 0]
        for dev in ["cpu:0"]:
            input = (
                torch.tensor([translate[0], translate[1], translate[2],
                              axis[0], axis[1], axis[2]], dtype=torch.double,
                             requires_grad=True, device=dev),
                #torch.tensor([1.8346128054e-03, 1.8701939553e-04, 3.7582265213e-04,
                #              -2.5503468351e-04, 2.3388580885e-03, 3.4152403714e-06],
                # dtype=torch.double, requires_grad=True, device=dev),
                #torch.rand(6, dtype=torch.double, requires_grad=True, device=dev),
                #torch.rand(1, 3, dtype=torch.double,
                #requires_grad=False, device=dev)
                torch.tensor([[1.0, 0, 0]], dtype=torch.double,
                             requires_grad=False, device=dev)
            )
            torch.autograd.gradcheck(transform, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)

    def _test_quat_t_transform(self):
        """Gradcheck the ExpRtTransform layer.
        """
        transform = se3.QuatRtTransform.apply
        torch.manual_seed(10)
        import math

        angle = math.pi / 3

        import quaternion
        quat = quaternion.from_euler_angles(45, 45, 45)

        axis = torch.Tensor([1.0, 1.0, 1.0])
        axis /= axis.norm(2)
        axis *= angle

        for dev in ["cpu:0"]:
            input = (torch.tensor([14, 4.2, 2, quat[0], quat[1], quat[2], quat[3]],
                                  dtype=torch.double, requires_grad=True, device=dev),
                     torch.rand(1, 3, dtype=torch.double,
                                requires_grad=False, device=dev))
            torch.autograd.gradcheck(transform, input, eps=1e-6, atol=1e-4,
                                     raise_exception=True)
