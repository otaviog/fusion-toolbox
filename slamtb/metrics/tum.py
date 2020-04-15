import torch
import numpy
import tenviz.pose

import slamtb.thirdparty.tumrgbd as _tumrgbd


def align_trajectories(trajectory_true, trajectory_pred, scale=1):
    xyz_true = numpy.matrix([numpy.array(rt_cam.center)
                             for rt_cam in trajectory_true.values()]).transpose()
    xyz_pred = numpy.matrix([numpy.array(rt_cam.center)*scale
                             for rt_cam in trajectory_pred.values()]).transpose()
    rot, trans, _ = _tumrgbd.align(xyz_true, xyz_pred)

    trans = torch.from_numpy(trans).double().squeeze()
    rot = torch.from_numpy(rot).double()

    align = tenviz.pose.Pose.from_rotation_matrix_translation(
        rot, trans).to_matrix().inverse()

    return {time: rt_cam.transform(align)
            for time, rt_cam in trajectory_pred.items()}


def absolute_translational_error(trajectory_true, trajectory_pred, scale=1):
    """Computes translational error. Code is a wrapper around TUM-RGBD source.
    """

    xyz_true = numpy.matrix([rt_cam.center.cpu().numpy()
                             for rt_cam in trajectory_true.values()]).transpose()
    xyz_pred = numpy.matrix([rt_cam.center.cpu().numpy()*scale
                             for rt_cam in trajectory_pred.values()]).transpose()

    _, _, trans_error = _tumrgbd.align(xyz_true, xyz_pred)

    return torch.from_numpy(trans_error)


def _tumrgbd_evaluate_trajectory(trajectory_true, trajectory_pred, max_pairs,
                                 fixed_delta, delta, delta_unit,
                                 offset, scale):
    return _tumrgbd.evaluate_trajectory(
        {time: rt_cam.matrix.cpu().numpy()
         for time, rt_cam in trajectory_true.items()},
        {time: rt_cam.matrix.cpu().numpy()
         for time, rt_cam in trajectory_pred.items()},
        max_pairs, fixed_delta, delta, delta_unit, offset, scale)


def relative_translational_error(trajectory_true, trajectory_pred, max_pairs=10000,
                                 fixed_delta=False, delta=1.0, delta_unit='s',
                                 offset=0.0, scale=1.0):
    """Computes translational error. Code is a wrapper around TUM-RGBD source.

    Args:

         max_pairs (int): maximum number of pose comparisons (default:
          10000, set to zero to disable downsampling).

         fixed_delta (bool): only consider pose pairs that have a
          distance of delta delta_unit (e.g., for evaluating the drift
          per second/meter/radian).

         delta (float): delta for evaluation (default: 1.0).

         delta_unit (str): unit of delta. Options: \'s\' for seconds,
          \'m\' for meters, \'rad\' for radians, \'f\' for frames;
          default: \'s\')'. Default is 's'.

         offset (float): time offset between ground-truth and
          estimated trajectory (default: 0.0).

         scale (float): scaling factor for the estimated trajectory
          (default: 1.0).

    """

    result = _tumrgbd_evaluate_trajectory(trajectory_true, trajectory_pred,
                                          max_pairs, fixed_delta, delta, delta_unit,
                                          offset, scale)

    rot_error = numpy.array(result)[:, 4]

    return torch.from_numpy(rot_error)


def relative_rotational_error(trajectory_true, trajectory_pred, max_pairs=10000,
                              fixed_delta=False, delta=1.0, delta_unit='s',
                              offset=0.0, scale=1.0):
    """Computes rotational error. Code is a wrapper around TUM-RGBD source.

    Args:

         max_pairs (int): maximum number of pose comparisons (default:
          10000, set to zero to disable downsampling).

         fixed_delta (bool): only consider pose pairs that have a
          distance of delta delta_unit (e.g., for evaluating the drift
          per second/meter/radian).

         delta (float): delta for evaluation (default: 1.0).

         delta_unit (str): unit of delta. Options: \'s\' for seconds,
          \'m\' for meters, \'rad\' for radians, \'f\' for frames;
          default: \'s\')'. Default is 's'.

         offset (float): time offset between ground-truth and
          estimated trajectory (default: 0.0).

         scale (float): scaling factor for the estimated trajectory
          (default: 1.0).

    """

    result = _tumrgbd_evaluate_trajectory(trajectory_true, trajectory_pred,
                                          max_pairs, fixed_delta, delta, delta_unit,
                                          offset, scale)

    rot_error = numpy.array(result)[:, 5]

    return torch.from_numpy(rot_error)
