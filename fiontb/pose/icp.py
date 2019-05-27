from open3d import (registration_colored_icp, ICPConvergenceCriteria,
                    voxel_down_sample, registration_icp,
                    TransformationEstimationPointToPlane)

import numpy as np
from fiontb import from_open3d, to_open3d
from fiontb.camera import RTCamera


def estimate_icp(live_pcl, model_pcl, compute_normals=False):
    live_pcl = live_pcl.to_open3d(compute_normals=compute_normals)
    model_pcl = model_pcl.to_open3d(compute_normals=compute_normals)

    result = registration_colored_icp(
        live_pcl, model_pcl, 0.5, np.eye(4),
        ICPConvergenceCriteria(relative_fitness=1e-6,
                               relative_rmse=1e-6, max_iteration=100))

    return RTCamera(result.transformation)


def estimate_icp_geo(live_pcl, model_pcl):
    live_pcl = live_pcl.to_open3d(compute_normals=True)
    model_pcl = model_pcl.to_open3d(compute_normals=True)

    result = registration_icp(
        live_pcl, model_pcl, 0.02, np.eye(4),
        TransformationEstimationPointToPlane())

    return RTCamera(result.transformation)
