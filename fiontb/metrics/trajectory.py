import torch


def ate_rmse(trajectory_true, trajectory_pred):
    # TODO: Align using the method of Horn. TUM-RGBD have an implementation.

    true2pred_space = trajectory_pred[0] @ trajectory_true[0].inverse()

    xyz_true = []
    xyz_pred = []
    for traj_true, traj_pred in zip(trajectory_true[1:], trajectory_pred[1:]):
        traj_true = true2pred_space @ traj_true

        xyz_true.append(traj_true.matrix[:3, 3])
        xyz_pred.append(traj_pred.matrix[:3, 3])

    xyz_true = torch.stack(xyz_true)
    xyz_pred = torch.stack(xyz_pred)

    rmse = (xyz_true - xyz_pred).norm(dim=1)
    rmse = rmse.mean()

    return rmse
