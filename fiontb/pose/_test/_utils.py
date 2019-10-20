import torch

from fiontb.metrics import absolute_translational_error
from fiontb.camera import RTCamera


def evaluate(dataset, relative_rt, other_frame_index):
    ate = absolute_translational_error({0.0: dataset.get_info(0).rt_cam,
                                        1.0: dataset.get_info(other_frame_index).rt_cam},
                                       {0.0: RTCamera(torch.eye(4)), 1.0: RTCamera(relative_rt.cpu())})
    
    print("ATE-RMSE: ", ate.mean().item())
