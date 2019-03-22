"""Custom RGB-D SLAM dataset format developed by Fusion toolbox
"""
from pathlib import Path
import json

import cv2
from tqdm import tqdm
import numpy as np

from .datatype import Snapshot
from fiontb.camera import KCamera, RTCamera


def _array_to_json(array):
    if isinstance(array, list):
        return array
    return array.tolist()


def kcam_to_json(kcam):
    return {'matrix': _array_to_json(kcam.matrix),
            'undist_coeff': _array_to_json(kcam.undist_coeff),
            'image_size': kcam.image_size,
            'depth_radial_distortion': kcam.depth_radial_distortion}


def write_ftb(base_path, dataset, prefix="frame_", max_frames=None):
    base_path = Path(base_path)

    base_path.mkdir(parents=True, exist_ok=True)

    frame_infos = []

    if max_frames is not None:
        max_frames = min(max_frames, len(dataset))
    else:
        max_frames = len(dataset)

    for i in tqdm(range(max_frames)):
        snap = dataset[i]

        frame_dict = {
            'kcam': kcam_to_json(snap.kcam),
            'depth_scale': snap.depth_scale,
            'depth_bias': snap.depth_bias,
            'depth_max': snap.depth_max}

        depth_file = "{}{:05d}_depth.png".format(prefix, i)

        cv2.imwrite(str(base_path / depth_file),
                    snap.depth_image.astype(np.uint16))
        frame_dict['depth_image'] = depth_file

        if snap.rgb_image is not None:
            rgb_file = "{}{:05d}_rgb.png".format(prefix, i)
            cv2.imwrite(str(base_path / rgb_file),
                        snap.rgb_image)
            frame_dict['rgb_image'] = rgb_file

        if snap.fg_mask is not None:
            # cv2.imwrite(str(base_path / "{}_{:05d}_mask.png".format(prefix, i)),
            # snap.fg_mask)
            pass  # TODO

        if snap.rt_cam is not None:
            frame_dict['rt_cam'] = _array_to_json(snap.rt_cam.matrix)

        if snap.rgb_kcam:
            frame_dict['rgb_kcam'] = kcam_to_json(snap.rgb_kcam)

        if snap.timestamp is not None:
            frame_dict['timestamp'] = snap.timestamp

        frame_infos.append(frame_dict)

    with open(str(base_path / "frame-info.json"), 'w') as stream:
        json.dump({'root': frame_infos}, stream, indent=1)


def kcam_from_json(js):
    return KCamera(np.array(js['matrix']),
                   js['undist_coeff'],
                   js['depth_radial_distortion'],
                   js['image_size'])


class FTBDataset:
    def __init__(self, frame_infos, base_path, ground_truth_model_path=None):
        self.frame_infos = frame_infos
        self.base_path = base_path
        self.ground_truth_model_path = ground_truth_model_path

    def __getitem__(self, idx):
        frame_info = self.frame_infos[idx]
        depth_image = cv2.imread(
            str(self.base_path / frame_info['depth_image']), cv2.IMREAD_ANYDEPTH)

        rgb_image = None
        if 'rgb_image' in frame_info:
            rgb_image = cv2.imread(
                str(self.base_path / frame_info['rgb_image']))

        fg_mask = None
        if 'fg_mask' in frame_info:
            fg_mask = cv2.imread(str(self.base_path / frame_info['fg_mask']))

        rt_cam = None
        if 'rt_cam' in frame_info:
            rt_cam = RTCamera(np.array(frame_info['rt_cam']))

        rgb_kcam = None
        if 'rgb_kcam' in frame_info:
            rgb_kcam = kcam_from_json(frame_info['rgb_kcam'])

        timestamp = frame_info.get('timestamp', None)

        snap = Snapshot(
            depth_image, kcam=kcam_from_json(frame_info['kcam']),
            depth_scale=frame_info['depth_scale'], depth_bias=frame_info['depth_bias'],
            depth_max=frame_info['depth_max'], rgb_image=rgb_image,
            rt_cam=rt_cam, rgb_kcam=rgb_kcam, fg_mask=fg_mask, timestamp=timestamp)

        return snap

    def __len__(self):
        return len(self.frame_infos)


def load_ftb(base_path, ground_truth_model_path=None):
    base_path = Path(base_path)
    with open(str(base_path / "frame-info.json"), 'r') as stream:
        frame_infos = json.load(stream)['root']

    return FTBDataset(frame_infos, base_path, ground_truth_model_path)
