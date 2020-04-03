"""Creation of registration algorithms with good "factory defaults".
"""

import math

from .icp import ICPOdometry
from .multiscale import MultiscaleRegistration


def create_multiscale_odometry(geom_weight=10.0, feat_weight=1.0):
    """Create a odometry object.

    Args:

        geom_weight (float): weight of the geometric term.

        feat_weight (float): weight of the feature term.

    Returns: (:obj:`MultiscaleRegistration`):
        An odomety good parameters.
    """
    params1 = {
        'geom_weight': geom_weight,
        'feat_weight': feat_weight,
        'distance_threshold': 1.5,
        'normals_angle_thresh': math.pi,
        'feat_residual_thresh': 5
    }

    params2 = {
        'geom_weight': geom_weight,
        'feat_weight': feat_weight,
        'distance_threshold': 1,
        'normals_angle_thresh': math.pi/2,
        'feat_residual_thresh': 2
    }

    params3 = {
        'geom_weight': geom_weight,
        'feat_weight': feat_weight,
        'distance_threshold': 1,
        'normals_angle_thresh': math.pi/4,
        'feat_residual_thresh': 2
    }

    return MultiscaleRegistration([
        (1.0, ICPOdometry(5, **params1)),
        (0.5, ICPOdometry(10, **params2)),
        (0.5, ICPOdometry(20, **params3))])
