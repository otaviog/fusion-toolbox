"""TUM-RGBD dataset parsing.
"""
from pathlib import Path

import numpy as np
import quaternion
import cv2

from fiontb.camera import RTCamera, KCamera
from fiontb.frame import Frame, FrameInfo

KCAMERA = KCamera.from_params(flen_x=525.0, flen_y=-525.0,
                              center_point=(319.5, 239.5))
DEFAULT_DEPTH_SCALE = 1.0/5000.0


class TUMRGBDDataset:
    """TUM indexed dataset.
    """

    def __init__(self, base_path, depths, rgbs, depth_rgb_assoc,
                 depth_gt_traj, depth_scale=DEFAULT_DEPTH_SCALE):
        self.base_path = base_path
        self.depths = depths
        self.rgbs = rgbs
        self.depth_rgb_assoc = depth_rgb_assoc
        self.depth_gt_traj = depth_gt_traj
        self.depth_scale = depth_scale

    def __getitem__(self, idx):
        depth_ts, rgb_ts = self.depth_rgb_assoc[idx]

        depth_img = cv2.imread(str(self.base_path / self.depths[depth_ts][0]),
                               cv2.IMREAD_ANYDEPTH)
        rgb_img = cv2.imread(str(self.base_path / self.rgbs[rgb_ts][0]))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        info = self.get_info(idx)

        return Frame(info, depth_img.astype(np.int32), rgb_img)

    def __len__(self):
        return len(self.depth_rgb_assoc)

    def get_info(self, idx):
        depth_ts, _ = self.depth_rgb_assoc[idx]
        rt_cam = self.depth_gt_traj[depth_ts]
        info = FrameInfo(kcam=KCAMERA, depth_scale=self.depth_scale, depth_bias=0.0,
                         timestamp=depth_ts, rt_cam=rt_cam)

        return info


def read_trajectory(gt_filepath, inv_y=False):
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

            rot_mtx = quaternion.as_rotation_matrix(
                np.quaternion(qw, qx, qy, qz))

            cam_mtx = np.eye(4)
            cam_mtx[0:3, 0:3] = rot_mtx
            cam_mtx[0:3, 3] = [tx, ty, tz]

            if inv_y:
                cam_mtx[:3, 1] *= -1
            gt_traj[timestamp] = RTCamera(cam_mtx)

    return gt_traj


def load_tumrgbd(base_path, assoc_offset=0.0, assoc_max_diff=0.2, depth_scale=None):
    """Loads the tumrgbd

    Args:

        base_path (str or :obj:`Path`): dataset base path.

        assoc_offset (float): Timestamp association offset.

        assoc_max (float): Maximum timestamp difference between two
         consecutive frames.

        depth_scale (float): TUM-RGBD dataset format multiplies depth
         values by 1.0/5000.0, non-standard may tweek this by
         providing its own value.

    Returns:
        (:obj:`TUMRGBDDataset`): Indexed snapshot dataset.
    """

    from fiontb.thirdparty.tumrgbd import associate, read_file_list

    base_path = Path(base_path)

    rgbs = read_file_list(str(base_path / "rgb.txt"))
    depths = read_file_list(str(base_path / "depth.txt"))
    gt_traj = read_trajectory(str(base_path / "groundtruth.txt"), inv_y=True)

    depth_rgb = associate(depths, rgbs, assoc_offset, assoc_max_diff)
    depth_gt = associate(depths, gt_traj, assoc_offset, assoc_max_diff)

    depth_traj = {}
    for depth_ts, gt_ts in depth_gt:
        depth_traj[depth_ts] = gt_traj[gt_ts]

    if depth_scale is None:
        depth_scale = DEFAULT_DEPTH_SCALE
    return TUMRGBDDataset(base_path, depths, rgbs, depth_rgb, depth_traj,
                          depth_scale)


def write_trajectory(filepath, rt_cams):
    """Write trajectory in the TUM-RGBD Format: timestamp pos and
    quaternion.

    Args:

        filepath (str): Output file

        rt_cams (Dict[(float, RTCamera)]): List of timestamps and RTCameras.

    """

    with open(filepath, 'w') as gt_txt:
        for timestamp, rt_cam in rt_cams.items():
            pos = rt_cam.matrix[0:3, 3]
            rot = rt_cam.matrix[0:3, 0:3]

            try:
                rot = quaternion.from_rotation_matrix(rot)
            except:
                rot = quaternion.from_euler_angles(0.0, 0.0, 0.0)

            gt_txt.write('{} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f} {:.9f}\n'.format(
                timestamp,
                pos[0], pos[1], pos[2],
                rot.x, rot.y, rot.z, rot.w))
