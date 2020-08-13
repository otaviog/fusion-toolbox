"""Registration result structures and utility functions.
"""

import math

import torch


class RegistrationResult:
    """Result information from registration algorithms.

    Attributes:

        transform (:obj:`torch.Tensor`): Result transformation (4 x 4) float64 matrix.

        information (:obj:`torch.Tensor`): Result information matrix (6 x 6) float64 matrix.

        residual (float): Final optimization residual.

        match_ratio (float): Final ratio of matched source points on
         the target.

        source_size (int, optional): Source geometry point count.

        target_size (int, optional): Target geometry point count.

    """

    def __init__(self, transform=None, information=None,
                 residual=math.inf, match_ratio=0, source_size=None, target_size=None):
        self.transform = transform
        self.information = information
        self.residual = residual
        self.match_ratio = match_ratio
        self.source_size = source_size
        self.target_size = target_size

    def __str__(self):
        return (f"RegistrationResult with: "
                + f"transform = {self.transform} "
                + f"information = {self.information} "
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
         information has value higher than this, then it'll declare a ill
         estimation. Seem in ElasticFusion.

        match_ratio_threshold (float): Results with lower point
         matching ratio are declared as ill estimations.

    """

    def __init__(self, match_ratio_threshold=6e-1, covariance_max_threshold=1e-04,
                 residual_threshhold=1e-4):

        self.match_ratio_threshold = match_ratio_threshold
        self.covariance_max_threshold = covariance_max_threshold
        self.residual_threshhold = residual_threshhold

    def __call__(self, result):
        if result.residual > self.residual_threshhold:
            return False

        if result.information is not None:
            eigvals, _ = torch.eig(result.information.inverse(), eigenvectors=False)
            if torch.any(eigvals[:, 0] > self.covariance_max_threshold):
                return False

        if result.match_ratio < self.match_ratio_threshold:
            return False

        return True

    def __str__(self):
        return (f"RegistrationVerifier with: match_ratio_threshold = {self.match_ratio_threshold} "
                + f"covariance_max_threshold = {self.covariance_max_threshold}")

    def __repr__(self):
        return str(self)
