"""Registration result structures and utility functions.
"""

import math

import torch


class RegistrationResult:
    """Result information from registration algorithms.

    Attributes:

        transform (:obj:`torch.Tensor`): Result transformation (4 x 4) float64 matrix.

        hessian (:obj:`torch.Tensor`): Result hessian matrix (6 x 6) float64 matrix.

        residual (float): Final optimization residual.

        match_ratio (float): Final ratio of matched source points on
         the target.

    """

    def __init__(self, transform=None, hessian=None,
                 residual=math.inf, match_ratio=0):
        self.transform = transform
        self.hessian = hessian
        self.residual = residual
        self.match_ratio = match_ratio

    def __str__(self):
        return (f"ICPResult with: "
                + f"transform = {self.transform} "
                + f"hessian = {self.hessian} "
                + f"residual = {self.residual} "
                + f"match_ratio = {self.match_ratio}")

    def __repr__(self):
        return str(self)


class RegistrationVerifier:
    """A basic heuristic for verifying ICP results.

    The checking is done by its __call__ operator.


    Attributes:

        residual_threshhold (float): Results with residuals lower than
         this are declared as good estimations.

        covariance_max_threshold (float): If any covariance element of
         hessian has value higher than this, then it'll declare a ill
         estimation. Seem in ElasticFusion.

        match_ratio_threshold (float): Results with lower point
         matching ratio are declared as ill estimations.

    """

    def __init__(self, match_ratio_threshold=6e-1, covariance_max_threshold=1e-04,
                 residual_threshhold=1e-4):

        self.match_ratio_threshold = match_ratio_threshold
        self.covariance_max_threshold = covariance_max_threshold
        self.residual_threshhold = residual_threshhold

    def __call__(self, icp_result):
        if icp_result.residual < self.residual_threshhold:
            return True

        if icp_result.hessian is not None:
            covariance = icp_result.hessian.lu()[0].inverse()
            if torch.any(covariance > self.covariance_max_threshold):
                return False

        if icp_result.match_ratio < self.match_ratio_threshold:
            return False

        return True

    def __str__(self):
        return (f"ICPVerifier with: match_ratio_threshold = {self.match_ratio_threshold} "
                + f"covariance_max_threshold = {self.covariance_max_threshold}")

    def __repr__(self):
        return str(self)
