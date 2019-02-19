"""TUM-RGBD dataset parsing.
"""
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion
import cv2

from fiontb.camera import RTCamera, KCamera

from .datatype import Snapshot

KCAMERA = KCamera.create_from_params(flen_x=525.0, flen_y=525.0,
                                     center_point=(319.5, 239.5))
_FACTOR = 5000


class TUMRGBDDataset:
    def __init__(self, base_path, depths, rgbs, depth_rgb_assoc, depth_gt_traj):
        self.base_path = base_path
        self.depths = depths
        self.rgbs = rgbs
        self.depth_rgb_assoc = depth_rgb_assoc
        self.depth_gt_traj = depth_gt_traj

    def __getitem__(self, idx):
        depth_ts, rgb_ts = self.depth_rgb_assoc[idx]

        depth_img = cv2.imread(str(self.base_path / self.depths[depth_ts][0]),
                               cv2.IMREAD_ANYDEPTH)
        rgb_img = cv2.imread(str(self.base_path / self.rgbs[rgb_ts][0]))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        return Snapshot(
            depth_img, kcam=KCAMERA, rgb_image=rgb_img,
            depth_scale=1.0/_FACTOR,
            rt_cam=RTCamera(self.depth_gt_traj[depth_ts]),
            timestamp=depth_ts)

    def __len__(self):
        return len(self.depth_rgb_assoc)


def _read_groundtruth(gt_filepath):
    gt_traj = {}
    with open(gt_filepath, 'r') as stream:
        while True:
            line = stream.readline()
            if line == "":
                break
            line = line.strip()
            if line.startswith("#"):
                continue

            entry = map(float, line.split())
            timestamp, tx, ty, tz, qx, qy, qz, qw = entry

            rot_mtx = Quaternion(qw, qx, qy, qz).transformation_matrix
            trans_mtx = np.eye(4)
            trans_mtx[0:3, 3] = [tx, ty, tz]

            cam_mtx = np.matmul(trans_mtx, rot_mtx)
            gt_traj[timestamp] = cam_mtx

    return gt_traj


def load_tumrgbd(base_path, assoc_offset=0.0, assoc_max_diff=0.2):
    from fiontb.thirdparty.tumrgbd import associate, read_file_list

    base_path = Path(base_path)

    rgbs = read_file_list(str(base_path / "rgb.txt"))
    depths = read_file_list(str(base_path / "depth.txt"))
    gt_traj = _read_groundtruth(str(base_path / "groundtruth.txt"))

    depth_rgb = associate(depths, rgbs, assoc_offset, assoc_max_diff)
    depth_gt = associate(depths, gt_traj, assoc_offset, assoc_max_diff)

    depth_traj = {}
    for depth_ts, gt_ts in depth_gt:
        depth_traj[depth_ts] = gt_traj[gt_ts]

    return TUMRGBDDataset(base_path, depths, rgbs, depth_rgb, depth_traj)
