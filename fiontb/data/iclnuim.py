"""
ICL-NUIM Parsing and reading

https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
"""

from pathlib import Path
from collections import namedtuple

import numpy as np
import cv2
from natsort import natsorted

from fiontb.camera import KCamera, RTCamera
from fiontb.frame import Frame, FrameInfo

Entry = namedtuple("ICLNuimEntry", ["extr_cam", "depth_path", "rgb_path"])

CAM_INTRINSIC = KCamera(np.array(
    [[481.20,	0,	319.50],
     [0,	480.00,	239.50],
     [0,	0,	1]]), depth_radial_distortion=False)


def undistort_depth(depth_image, kcam_matrix):
    xs, ys = np.meshgrid(np.arange(depth_image.shape[1]),
                         np.arange(depth_image.shape[0]))

    points = np.dstack(
        [xs, ys, depth_image])
    points = points.reshape((-1, 3, 1))

    xyz_coords = points[:, 0:2]
    xyz_coords = np.insert(xyz_coords, 2, 1.0, axis=1)
    xyz_coords = np.matmul(np.linalg.inv(
        kcam_matrix), xyz_coords)

    depths = points[:, 2, 0]
    depths = (depths /
              np.sqrt(np.power(xyz_coords[:, 0:2, 0], 2).sum(1) + 1))

    return depths.reshape(depth_image.shape)


class ICLNuim:
    def __init__(self, trajectory, ground_truth_model_path=None):
        self.trajectory = trajectory
        self.ground_truth_model_path = ground_truth_model_path

    def __getitem__(self, idx):
        entry = self.trajectory[idx]

        color_img = cv2.imread(str(entry.rgb_path))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        with open(str(entry.depth_path)) as file:
            depth_image = [float(elem) for elem in file.read().split()]

        depth_image = np.array(depth_image)
        depth_image[depth_image > 1e3] = -1
        depth_image = depth_image.reshape(color_img.shape[0:2])
        depth_image = undistort_depth(depth_image,
                                      CAM_INTRINSIC.matrix)

        info = FrameInfo(CAM_INTRINSIC, depth_scale=1.0, depth_bias=0.0, depth_max=4500.0,
                         timestamp=idx, rt_cam=entry.extr_cam)
        frame = Frame(info, depth_image, color_img)
        return frame

    def __len__(self):
        return len(self.trajectory)


def _load_camera(cam_path):
    val_dict = {}
    with open(str(cam_path)) as cam_file:
        for line in cam_file.readlines():
            key, value = line.split('=')

            key = key.strip()
            value = value.strip()
            value = value.replace(";", '').replace("'", '')
            value = value.replace("[", '').replace("]", '')
            value = value.split(',')
            value = [float(v) for v in value]
            val_dict[key] = value

    zcol = np.array(val_dict["cam_dir"])
    zcol /= np.linalg.norm(zcol, 2)

    # We inverted ycol of ICL-Nuim to match other datasets
    ycol = -np.array(val_dict["cam_up"])
    ycol /= np.linalg.norm(ycol, 2)

    xcol = np.array(val_dict["cam_right"])
    xcol /= np.linalg.norm(xcol)

    rot_mtx = np.array([xcol, ycol, zcol]).T

    invx = np.eye(3)
    invx[0, 0] = -1.0

    rot_mtx = np.matmul(invx, rot_mtx)

    pos = np.array(val_dict["cam_pos"])
    pos[0] *= -1

    return RTCamera.create_from_params(pos, rot_mtx)


def _load_sim_camera(filepath):
    with open(str(filepath), 'r') as stream:
        lines = stream.readlines()

    sim_traj = []
    for i in range(0, len(lines), 4):
        row0 = [float(elem) for elem in lines[i].split()]
        row1 = [float(elem) for elem in lines[i + 1].split()]
        row2 = [float(elem) for elem in lines[i + 2].split()]

        cam_matrix = np.vstack([row0, row1, row2, np.array([0, 0, 0, 1])])

        sim_traj.append(RTCamera(cam_matrix))

    return sim_traj


def load_icl_nuim(base_path, sim_traj_filepath=None, ground_truth_model_path=None):
    """Loads a ICL-NUIM scene as an indexed Snapshot dataset.

    Args:

        base_path (str): Base scene path, i.e.,
         "ICL-NUIM/living_room_traj0_loop"

    Returns: (:obj:ICLNuim):

        Snapshot indexed dataset.

    """

    base_path = Path(base_path)

    img_glob = base_path.glob("scene_*.png")
    img_glob = natsorted(img_glob, key=lambda key: str(key))

    trajectory = []
    gt_traj = None
    if sim_traj_filepath is not None:
        gt_traj = _load_sim_camera(sim_traj_filepath)
        img_glob = img_glob[2:]

    for i, img_path in enumerate(img_glob):
        depth_path = img_path.with_suffix('.depth')
        cam_info_path = img_path.with_suffix('.txt')

        if gt_traj is None:
            cam_ext = _load_camera(cam_info_path)
        else:
            cam_ext = gt_traj[i]

        entry = Entry(cam_ext, depth_path, img_path)
        trajectory.append(entry)

    return ICLNuim(trajectory, ground_truth_model_path)
