"""Parser for the SceneNN RGB-D dataset from .oni files.
"""

import numpy as np
import onireader

from fiontb.camera import KCamera, RTCamera
from .datatype import Snapshot


KINECT2_KCAM = KCamera(np.array([[356.769928, 0.0, 251.563446],
                                 [0.0, 430.816498, 237.563446],
                                 [0.0, 0.0, 1.0]]))

ASUS_KCAM = KCamera(np.array([[544.47329, 0.0, 320],
                              [0.0, 544.47329, 240],
                              [0.0, 0.0, 1.0]]))


class SceneNN:
    """(Almost)-Indexed snapshot dataset for SceneNN. Note due to oni
    playback, this class always advanced one frame no matter which
    indexed is passed. Use :func:`rewind` to go to the begining.

    """

    def __init__(self, oni_filepath, trajectory, k_cam, ground_truth_model_path):
        self._oni_filepath = oni_filepath
        self.rewind()

        self.trajectory = trajectory
        self.k_cam = k_cam

        self.first_frame_id = None
        self.last_idx = None
        self.cache = None

        self.ground_truth_model_path = ground_truth_model_path

    def rewind(self):
        """Rewinds the data to the begining frames.
        """
        # self.ni_dev.seek doesn't work
        self.ni_dev = None
        self.ni_dev = onireader.Device()
        self.ni_dev.open(str(self._oni_filepath))

    def _getnext_pair(self):
        depth_img, depth_ts, _ = self.ni_dev.readDepth()
        rgb_img, rgb_ts, _ = self.ni_dev.readColor()

        k_time_diff = 33000
        diff = abs(rgb_ts - depth_ts)

        while diff > k_time_diff:
            if rgb_ts > depth_ts:
                depth_img, depth_ts, _ = self.ni_dev.readDepth()
            else:
                rgb_img, rgb_ts, _ = self.ni_dev.readColor()

            diff = abs(rgb_ts - depth_ts)
            print("Skiping rgb {} and depth {}".format(rgb_ts, depth_ts))
        return depth_img, rgb_img, depth_ts

    def __getitem__(self, idx):
        # pylint: disable=unused-variable
        if self.last_idx != idx:
            self.cache = self._getnext_pair()
            self.last_idx = idx

        depth_img, rgb_img, depth_ts = self.cache

        rt_mtx = self.trajectory[idx]

        return Snapshot(depth_img, kcam=self.k_cam, rgb_image=rgb_img,
                        rt_cam=RTCamera(rt_mtx), depth_scale=0.001,
                        timestamp=depth_ts)

    def __len__(self):
        return len(self.trajectory)


def load_scenenn(oni_filepath, traj_filepath, k_cam_dev='asus', ground_truth_model_path=None):
    trajectory = []
    with open(traj_filepath, 'r') as file:
        while True:
            line = file.readline()
            if line == "":
                break
            curr_entry = []
            for _ in range(4):
                line = file.readline()
                curr_entry.append([float(elem) for elem in line.split()])
            rt_mtx = np.array(curr_entry)  # cam space to world space

            assert rt_mtx.shape == (4, 4)
            trajectory.append(rt_mtx)

    k_cams = {'asus': ASUS_KCAM, 'kinect2': KINECT2_KCAM}

    if k_cam_dev not in k_cams:
        raise RuntimeError("Undefined {} camera intrinsics. Use: {}".format(
            k_cam_dev, k_cam_dev.keys()))

    return SceneNN(oni_filepath, trajectory, k_cams[k_cam_dev], ground_truth_model_path)
