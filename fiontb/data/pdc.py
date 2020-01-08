from pathlib import Path

import yaml
import cv2
import quaternion
import numpy as np
import torch
from tqdm import tqdm

from fiontb.frame import Frame, FrameInfo
from fiontb.camera import KCamera, RTCamera


class PDCDataset:
    def __init__(self, base_dir, kcam, depth_scale, pose_dict):
        self.base_dir = base_dir
        self.kcam = kcam
        self.depth_scale = depth_scale
        self.pose_dict = pose_dict
        self._keys = list(pose_dict.keys())

    def __getitem__(self, idx):
        pose = self.pose_dict[self._keys[idx]]

        quat = pose['camera_to_world']['quaternion']
        qw, qx, qy, qz = quat['w'], quat['x'], quat['y'], quat['z']

        trans = pose['camera_to_world']['translation']
        tx, ty, tz = trans['x'], trans['y'], trans['z']
        rot_mtx = quaternion.as_rotation_matrix(
            np.quaternion(qw, qx, qy, qz))

        cam_mtx = np.eye(4)
        cam_mtx[0:3, 0:3] = rot_mtx
        cam_mtx[0:3, 3] = [tx, ty, tz]

        timestamp = pose['timestamp']

        info = FrameInfo(kcam=self.kcam, depth_scale=self.depth_scale,
                         depth_bias=0.0, rt_cam=RTCamera(cam_mtx),
                         timestamp=timestamp)

        depth_file = self.base_dir / 'rendered_images' / \
            pose['depth_image_filename']
        rgb_file = self.base_dir / 'images' / pose['rgb_image_filename']

        depth_img = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
        rgb_img = cv2.imread(str(rgb_file))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        return Frame(info, depth_img.astype(np.int32), rgb_img)

    def __len__(self):
        return len(self.pose_dict)


def load_pdc(base_dir):
    base_dir = Path(base_dir)
    with open(base_dir / "images/pose_data.yaml") as pose_file:
        pose_dict = yaml.load(pose_file, Loader=yaml.FullLoader)

    with open(base_dir / "images/camera_info.yaml") as kcam_file:
        kcam_dict = yaml.load(kcam_file, Loader=yaml.FullLoader)

    kcam = KCamera(torch.tensor(
        kcam_dict['camera_matrix']['data']).reshape(3, 3))

    return PDCDataset(base_dir, kcam, 1.0/1000.0, pose_dict)


def write_pdc(base_dir, dataset, max_frames=None):
    base_dir = Path(base_dir)
    pose_dict = {}

    info = None
    if max_frames is None:
        max_frames = len(dataset)
    else:
        max_frames = min(len(dataset), max_frames)

    for idx in tqdm(range(max_frames)):
        frame = dataset[idx]

        if info is None:
            info = frame.info

        pos = frame.info.rt_cam.matrix[0:3, 3].tolist()
        quat = quaternion.from_rotation_matrix(
            frame.info.rt_cam.matrix[0:3, 0:3])

        depth_path = base_dir / "rendered_images" / \
            "{:06d}_depth.png".format(idx)
        rgb_path = base_dir / "images" / "{:06d}_rgb.png".format(idx)

        depth_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(depth_path), frame.depth_image.astype(np.uint16))
        cv2.imwrite(str(rgb_path), cv2.cvtColor(
            frame.rgb_image, cv2.COLOR_RGB2BGR))

        if frame.fg_mask is not None:
            mask_img = frame.fg_mask
        else:
            mask_img = np.ones(frame.depth_image.shape, dtype=np.uint8)

        mask_path = base_dir / "image_masks" / \
            "image_masks" / "{:06d}_mask.png".format(idx)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(mask_path), mask_img)


        pose_dict[idx] = {
            "camera_to_world": {
                "quaternion": {"w": quat.w, "x": quat.x, "y": quat.y, "z": quat.z},
                "translation": {"x": pos[0], "y": pos[1], "z": pos[2]},
            },
            "depth_image_filename": str(depth_path.name),
            "rgb_image_filename": str(rgb_path.name),
            "timestamp": frame.info.timestamp
        }

    with open(str(base_dir / "images" / "pose_data.yaml"), 'w') as pose_file:
        yaml.dump(pose_dict, pose_file)

    proj_matrix = torch.eye(4, dtype=info.kcam.matrix.dtype)
    proj_matrix[:3, :3] = info.kcam.matrix
    camera_dict = {
        "camera_matrix": {
            "cols": 3,
            "rows": 3,
            "data": info.kcam.matrix.flatten().tolist()
        },
        "distortion_coefficients": {
            "cols": 5,
            "rows": 1,
            "data": [0.0]*5
        },
        "distortion_model": "plumb_bob",
        "image_height": info.kcam.image_size[0],
        "image_width": info.kcam.image_size[1],
        "projection_matrix": {
            "cols": 4,
            "rows": 4,
            "data": proj_matrix.flatten().tolist()
        },
        "rectification_matrix": {
            "cols": 3,
            "rows": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        }
    }
    with open(str(base_dir / "images" / "camera_info.yaml"), 'w') as camera_file:
        yaml.dump(camera_dict, camera_file)
