import torch

class ICPResult:
    def __init__(self, transform, pose_matrix,
                 final_loss, best_loss,
                 final_match_ratio, best_match_ratio):
        self.transform = transform
        self.final_loss = final_loss
        self.best_loss = best_loss
        self.pose_matrix = pose_matrix
        self.final_match_ratio = final_match_ratio
        self.best_match_ratio = best_match_ratio

    def __str__(self):
        return (f"ICPResult with: final_loss = {self.final_loss} "
                + f"best_loss = {self.best_loss} "
                + f"final_match_ratio = {self.final_match_ratio} "
                + f"best_match_ratio = {self.best_match_ratio}")

    def __repr__(self):
        return str(self)


class ICPVerifier:
    def __init__(self, match_ratio_threshold=.6, covariance_max_threshold=1e-04):
        self.match_ratio_threshold = match_ratio_threshold
        self.covariance_max_threshold = covariance_max_threshold

    def __call__(self, icp_result):
        covariance = icp_result.pose_matrix.lu()[0].inverse()
        if torch.any(covariance > self.covariance_max_threshold):
            return False

        if icp_result.final_match_ratio < self.match_ratio_threshold:
            return False

        return True

    def __str__(self):
        return (f"ICPVerifier with: match_ratio_threshold = {self.match_ratio_threshold} "
                + f"covariance_max_threshold = {self.covariance_max_threshold}")

    def __repr__(self):
        return str(self)
