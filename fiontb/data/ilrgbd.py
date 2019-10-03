from pathlib import Path

import torch
import cv2
from natsort import natsorted
import numpy as np

from fiontb.frame import Frame, FrameInfo
from fiontb.camera import KCamera, RTCamera

from .trajectory import read_log_file_trajectory

ASUS_KCAM = KCamera(torch.tensor([[525, 0.0, 319.5],
                                  [0.0, 525, 239.5],
                                  [0.0, 0.0, 1.0]], dtype=torch.float))


class ILRGBDDataset:
    def __init__(self, depth_images, rgb_images, trajectory):
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.trajectory = trajectory

    def get_info(self, idx):
        rt_cam = self.trajectory[idx]

        return FrameInfo(ASUS_KCAM, 0.001, rt_cam=rt_cam)

    def __getitem__(self, idx):
        rgb_image = cv2.imread(str(self.rgb_images[idx]))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_image = cv2.imread(
            str(self.depth_images[idx]), cv2.IMREAD_ANYDEPTH).astype(np.int32)

        info = self.get_info(idx)

        return Frame(info, depth_image, rgb_image)

    def __len__(self):
        return min(len(self.rgb_images), len(self.depth_images))


def _read_json_trajectory(stream):
    import json

    trajectory = []
    traj_dict = json.load(stream)
    for node in traj_dict['nodes']:
        pose = torch.tensor(node['pose'])
        matrix = pose.reshape(4, 4).transpose(1, 0)
        matrix[:3, :3] = matrix[:3, :3].transpose(1, 0)
        if False:
            matrix = torch.eye(4)
            matrix[0, :3] = pose[:3]
            matrix[1, :3] = pose[3:6]
            matrix[2, :3] = pose[6:9]
            matrix[:3, 3] = pose[6:]

        trajectory.append(RTCamera(matrix))

    return trajectory


def _read_log_trajectory(stream):
    lines = stream.readlines()

    trajectory = []

    for i in range(0, len(lines), 5):
        matrix = [torch.tensor(list(map(float, lines[i + 1 + k].split())))
                  for k in range(4)]
        matrix = torch.stack(matrix)
        matrix[:3, :3] = matrix[:3, :3].transpose(1, 0)
        matrix[:3, 3] = -matrix[:3, 3]
        matrix[:3, 0] *= -1
        matrix[:3, 1] *= -1
        rt_cam = RTCamera(matrix)
        trajectory.append(rt_cam)

    return trajectory


def load_ilrgbd(base_dir, trajectory):
    rgb_images = (Path(base_dir) / "image").glob("*.jpg")
    rgb_images = natsorted(rgb_images, key=str)

    depth_images = (Path(base_dir) / "depth").glob("*.png")
    depth_images = natsorted(depth_images, key=str)

    with open(str(trajectory), 'r') as stream:
        trajectory = read_log_file_trajectory(stream)

    return ILRGBDDataset(depth_images, rgb_images, trajectory)
