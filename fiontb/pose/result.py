import torch


class ICPResult:
    def __init__(self, transform, hessian,
                 residual, match_ratio):
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


class ICPVerifier:
    def __init__(self, match_ratio_threshold=.6, covariance_max_threshold=1e-04):
        self.match_ratio_threshold = match_ratio_threshold
        self.covariance_max_threshold = covariance_max_threshold

    def __call__(self, icp_result):
        if icp_result.residual < 1e-4:
            return True

        if icp_result.hessian is None:
            return False

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
