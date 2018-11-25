"""Parser for the SceneNN RGB-D dataset from .oni files.
"""

import numpy as np
import onireader

from fusionkit.camera import KCamera, RTCamera
from .datatype import Snapshot


KINECT2_KCAM = KCamera(np.array([[356.769928, 0.0, 251.563446],
                                 [0.0, 430.816498, 237.563446],
                                 [0.0, 0.0, 1.0]]))

ASUS_KCAM = KCamera(np.array([[544.47329, 0.0, 320],
                              [0.0, 544.47329, 240],
                              [0.0, 0.0, 1.0]]))


class SceneNN:
    def __init__(self, oni_filepath, trajectory, k_cam):
        self.ni_dev = onireader.Device()
        self.ni_dev.open(str(oni_filepath))
        self.trajectory = trajectory
        self.k_cam = k_cam

    def __getitem__(self, idx):
        depth_img = self.ni_dev.readDepth()
        rgb_img = self.ni_dev.readColor() / 255.0

        rt_mtx = self.trajectory[idx]
        return Snapshot(depth_img, rgb_img, self.k_cam,
                        RTCamera(rt_mtx))

    def __len__(self):
        return len(self.ni_dev)


def load_scenenn(oni_filepath, traj_filepath, k_cam_dev='asus'):
    trajectory = []
    with open(traj_filepath, 'r') as file:

        while True:
            line = file.readline()
            if line == "":
                break
            curr_entry = []
            for i in range(4):
                line = file.readline()
                curr_entry.append([float(elem) for elem in line.split()])
            rt_mtx = np.array(curr_entry)  # cam space to world space

            trajectory.append(rt_mtx)

    k_cams = {'asus': ASUS_KCAM, 'kinect2': KINECT2_KCAM}

    if k_cam_dev not in k_cams:
        raise RuntimeError("Undefined {} camera intrinsics. Use: {}".format(
            k_cam_dev, k_cam_dev.keys()))

    return SceneNN(oni_filepath, trajectory, k_cams[k_cam_dev])
