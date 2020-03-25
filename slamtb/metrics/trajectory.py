"""Metrics for evaluating camera trajectories.
"""
import math
import torch

from slamtb.camera import RTCamera


def set_start_at_identity(trajectory):
    base = trajectory[next(iter(trajectory))].matrix.inverse()

    return {time: rt_cam.right_transform(base)
            for time, rt_cam in trajectory.items()}


def translational_difference(diff_matrix):
    """Returns the translation from a pose difference matrix.

    Args:

        diff_matrix (obj:`torch.Tensor`): [3x4] or [4x4] matrix.
    """
    return (diff_matrix[:3, 3]).norm(2)


def rotational_difference(diff_matrix):
    """Returns the rotation angle from a pose difference matrix.

    Args:

        diff_matrix (obj:`torch.Tensor`): [3x4] or [4x4] matrix.

    Returns: (float): Rotation angle.
    """
    return math.acos(min(1.0, max(-1.0, (torch.trace(diff_matrix[0:3, 0:3]) - 1.0)/2.0)))


def _ensure_matrix(cam_or_matrix):
    if isinstance(cam_or_matrix, RTCamera):
        return cam_or_matrix.matrix

    return cam_or_matrix


def absolute_translational_error(trajectory_true, trajectory_pred):
    """Computes the per trajectory absolute translational error as defined
    in Sturm, Jürgen, Nikolas Engelhard, Felix Endres, Wolfram
    Burgard, and Daniel Cremers. "A benchmark for the evaluation of
    RGB-D SLAM systems." In 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems, pp. 573-580. IEEE, 2012.

    Args:

        trajectory_true (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The ground truth trajectory.

        trajectory_pred (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The predicted ground truth trajectory.

    Returns: (obj:`torch.Tensor`): Per pose absolute translation
     error. Use `.mean().sqrt()` for having the RMSE

    """

    traj_errors = []
    for k, cam_true in trajectory_true.items():
        cam_pred = _ensure_matrix(trajectory_pred[k])
        cam_true = _ensure_matrix(cam_true)

        diff = translational_difference(cam_true.inverse() @ cam_pred)
        traj_errors.append(diff)

    return torch.tensor(traj_errors)


def absolute_rotational_error(trajectory_true, trajectory_pred):
    """Computes the per trajectory absolute rotational error as defined
    in Sturm, Jürgen, Nikolas Engelhard, Felix Endres, Wolfram
    Burgard, and Daniel Cremers. "A benchmark for the evaluation of
    RGB-D SLAM systems." In 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems, pp. 573-580. IEEE, 2012.

    Args:

        trajectory_true (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The ground truth trajectory.

        trajectory_pred (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The predicted ground truth trajectory.

    Returns: (obj:`torch.Tensor`): Per pose absolute rotational
     error. Use `.mean().sqrt()` for having the RMSE

    """

    traj_errors = []
    for k, cam_true in trajectory_true.items():
        cam_pred = trajectory_pred[k]
        cam_pred = _ensure_matrix(trajectory_pred[k])
        cam_true = _ensure_matrix(cam_true)

        diff = rotational_difference(cam_true.inverse() @ cam_pred)
        traj_errors.append(diff)

    return torch.tensor(traj_errors)


def _get_relative_matrices(trajectory_true, trajectory_pred):
    keys = list(trajectory_true.keys())
    diff_matrices = []

    for k in range(0, len(keys) - 1):
        i = keys[k]
        j = keys[k + 1]

        cam_true_i = _ensure_matrix(trajectory_true[i])
        cam_pred_i = _ensure_matrix(trajectory_pred[i])

        cam_true_j = _ensure_matrix(trajectory_true[j])
        cam_pred_j = _ensure_matrix(trajectory_pred[j])

        diff_matrix = ((cam_true_i.inverse() @ cam_true_j).inverse()
                       @ (cam_pred_i.inverse() @ cam_pred_j))

        diff_matrices.append(diff_matrix)

    return diff_matrices


def relative_translational_error(trajectory_true, trajectory_pred):
    """Computes the per trajectory relative translational error as defined
    in Sturm, Jürgen, Nikolas Engelhard, Felix Endres, Wolfram
    Burgard, and Daniel Cremers. "A benchmark for the evaluation of
    RGB-D SLAM systems." In 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems, pp. 573-580. IEEE, 2012.

    Args:

        trajectory_true (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The ground truth trajectory.

        trajectory_pred (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The predicted ground truth trajectory.

    Returns: (obj:`torch.Tensor`): Per pose relative translational
     error. Use `.mean().sqrt()` for having the RMSE

    """

    return torch.tensor(
        [translational_difference(diff_matrix)
         for diff_matrix in _get_relative_matrices(
            trajectory_true, trajectory_pred)])


def relative_rotational_error(trajectory_true, trajectory_pred):
    """Computes the per trajectory relative rotational error as defined
    in Sturm, Jürgen, Nikolas Engelhard, Felix Endres, Wolfram
    Burgard, and Daniel Cremers. "A benchmark for the evaluation of
    RGB-D SLAM systems." In 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems, pp. 573-580. IEEE, 2012.

    Args:

        trajectory_true (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The ground truth trajectory.

        trajectory_pred (Dict[float: :obj:`slamtb.camera.RTCamera`]):
         The predicted ground truth trajectory.

    Returns: (obj:`torch.Tensor`): Per pose relative rotational
     error. Use `.mean().sqrt()` for having the RMSE

    """
    return torch.tensor(
        [rotational_difference(diff_matrix)
         for diff_matrix in _get_relative_matrices(
            trajectory_true, trajectory_pred)])
