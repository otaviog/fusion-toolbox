import torch

from fiontb.metrics import absolute_translational_error
from fiontb.camera import RTCamera


def evaluate(dataset, relative_rt, other_frame_index):
    ate = absolute_translational_error([dataset.get_info(0).rt_cam,
                                        dataset.get_info(other_frame_index).rt_cam],
                                       [RTCamera(torch.eye(4)), RTCamera(relative_rt.cpu())])

    print("ATE-RMSE: ", ate.mean().item())
