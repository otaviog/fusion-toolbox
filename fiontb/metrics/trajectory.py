import quaternion
import torch

from tenviz.pose import Pose

from fiontb.camera import RTCamera, RigidTransform


def set_start_at_identity(trajectory):
    base = trajectory[next(iter(trajectory))].matrix.inverse()

    return {time: rt_cam.right_transform(base)
            for time, rt_cam in trajectory.items()}


def translational_difference(matrix_true, matrix_pred):
    if isinstance(matrix_true, RTCamera):
        matrix_true = matrix_true.matrix

    if isinstance(matrix_pred, RTCamera):
        matrix_pred = matrix_pred.matrix

    rt_true = RigidTransform(matrix_true)
    rt_pred = RigidTransform(matrix_pred)

    return (rt_true.translation() - rt_pred.translation()).norm(2)


def rotational_difference(matrix_true, matrix_pred):
    if isinstance(matrix_true, RTCamera):
        matrix_true = matrix_true.matrix

    if isinstance(matrix_pred, RTCamera):
        matrix_pred = matrix_pred.matrix

    pose_true = Pose.from_matrix(matrix_true)
    pose_pred = Pose.from_matrix(matrix_pred)

    gt_rot = quaternion.from_float_array(pose_true.get_quaternion())
    pred_rot = quaternion.from_float_array(pose_pred.get_quaternion())

    return (gt_rot - pred_rot).norm()


def _get_relative_matrices(trajectory_true, trajectory_pred):
    keys = list(trajectory_true.keys())
    rel_trues = []
    rel_preds = []

    for k in range(0, len(keys) - 1):
        i = keys[k]
        j = keys[k + 1]

        cam_true_i = trajectory_true[i]
        cam_pred_i = trajectory_pred[i]

        cam_true_j = trajectory_true[j]
        cam_pred_j = trajectory_pred[j]

        rel_trues.append(cam_true_i.matrix.inverse() @ cam_true_j.matrix)
        rel_preds.append(cam_pred_i.matrix.inverse() @ cam_pred_j.matrix)

    return rel_trues, rel_preds


def relative_translational_error(trajectory_true, trajectory_pred):
    rel_trues, rel_preds = _get_relative_matrices(
        trajectory_true, trajectory_pred)

    return torch.tensor([translational_difference(rel_true, rel_pred)
                         for rel_true, rel_pred in zip(rel_trues, rel_preds)])


def relative_rotational_error(trajectory_true, trajectory_pred):
    rel_trues, rel_preds = _get_relative_matrices(
        trajectory_true, trajectory_pred)

    return torch.tensor([rotational_difference(rel_true, rel_pred)
                         for rel_true, rel_pred in zip(rel_trues, rel_preds)])


def absolute_translational_error(trajectory_true, trajectory_pred):
    traj_errors = []
    for k, cam_true in trajectory_true.items():
        cam_pred = trajectory_pred[k]
        diff = translational_difference(cam_true, cam_pred)
        traj_errors.append(diff)

    return torch.tensor(traj_errors)


def absolute_rotational_error(trajectory_true, trajectory_pred):
    traj_errors = []
    for k, cam_true in trajectory_true.items():
        cam_pred = trajectory_pred[k]
        diff = rotational_difference(cam_true, cam_pred)
        traj_errors.append(diff)

    return torch.tensor(traj_errors)
