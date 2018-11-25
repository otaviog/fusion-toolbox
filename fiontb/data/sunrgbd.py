from pathlib import Path
from collections import namedtuple

import numpy as np
import cv2

from fusionkit.camera import Camera, IntrinsicCamera, ExtrinsicCamera
from .datatype import Snapshot

Entry = namedtuple(
    "SUNRGBDEntry", ["k_mtx", "rt_mtx", "depth_path", "rgb_path"])


class SUNRGBD:
    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __getitem__(self, idx):
        entry = self.trajectory[idx]
        color_img = cv2.imread(str(entry.rgb_path))
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        color_img = color_img/255

        depth_img = cv2.imread(str(entry.depth_path))

        depth_img = depth_img[:, :, 0]
        return Snapshot(depth_img, color_img, intr_cam=IntrinsicCamera(entry.k_mtx),
                        extr_cam=ExtrinsicCamera(entry.rt_mtx))


def _first_elem(gen):
    try:
        return next(iter(gen))
    except StopIteration:
        return None

def _load_frame_dir(path):
    rgb = _first_elem((path / "image").glob("*.jpg"))
    if rgb is None:
        import ipdb; ipdb.set_trace()
    depth = _first_elem((path / "depth").glob("*.png"))

    with open(path / "intrinsics.txt") as file:
        k_mtx = []
        for line in file.readlines():
            k_mtx.extend([float(entry) for entry in line.split()])
        k_mtx = np.array(k_mtx).reshape((3, 3))

    with open(_first_elem((path / "extrinsics").glob("*.txt"))) as file:
        rt_mtx = []
        for line in file.readlines():
            rt_mtx.extend([float(entry) for entry in line.split()])

        rt_mtx.extend([0.0, 0.0, 0.0, 1.0])
        rt_mtx = np.array(rt_mtx).reshape((4, 4))

    try:
        key = int(rgb.stem)
    except:
        key = rgb.stem
        
    return key, Entry(k_mtx, rt_mtx, depth, rgb)


def load_sunrgbd_scene(base_path):
    base_path = Path(base_path)
    frame_dirs = base_path.glob("*")
    frame_dirs = [frame_dir for frame_dir in frame_dirs
                  if frame_dir.is_dir()]
    # frame_dirs = sorted(
    #frame_dirs, key=lambda path: int(path.stem.split('_')[5]))

    frame_dirs = sorted(frame_dirs)
        
    trajectory = []

    for frame_dir in frame_dirs[0:300]:
        key, entry = _load_frame_dir(frame_dir)
        trajectory.append((key, entry))

    trajectory = sorted(trajectory, key=lambda e: e[0])

    trajectory = [entry for _, entry in trajectory]
    return SUNRGBD(trajectory)
