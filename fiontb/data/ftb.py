"""Custom RGB-D SLAM dataset format developed by Fusion toolbox.
"""
from pathlib import Path
import json

import cv2
from tqdm import tqdm
import numpy as np

from fiontb.frame import Frame, FrameInfo


def write_ftb(base_path, dataset, prefix="frame_", max_frames=None):
    """Write a dataset as a FTB directory.

    Args:

        base_path (str): Output directory path.

        dataset (object): Any fiontb dataset object.

        prefix (str): Image files prefix.

        max_frames (int, optional): Maximum number to frames to write.
    """

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    frames = []

    if max_frames is not None:
        max_frames = min(max_frames, len(dataset))
    else:
        max_frames = len(dataset)

    for i in tqdm(range(max_frames)):
        frame = dataset[i]
        info = frame.info
        frame_dict = {
            'info': info.to_json()
        }

        depth_file = "{}{:05d}_depth.png".format(prefix, i)

        cv2.imwrite(str(base_path / depth_file),
                    frame.depth_image.astype(np.uint16))
        frame_dict['depth_image'] = depth_file

        if frame.rgb_image is not None:
            rgb_file = "{}{:05d}_rgb.png".format(prefix, i)
            cv2.imwrite(str(base_path / rgb_file),
                        frame.rgb_image)
            frame_dict['rgb_image'] = rgb_file

        if frame.fg_mask is not None:
            # cv2.imwrite(str(base_path / "{}_{:05d}_mask.png".format(prefix, i)),
            # snap.fg_mask)
            pass  # TODO

        frames.append(frame_dict)

    with open(str(base_path / "frames.json"), 'w') as stream:
        json.dump({'root': frames}, stream, indent=1)


class FTBDataset:
    def __init__(self, frames_json, base_path, ground_truth_model_path=None):
        self.frames_json = frames_json
        self.base_path = base_path
        self.ground_truth_model_path = ground_truth_model_path

    def __getitem__(self, idx):
        frame_json = self.frames_json[idx]
        depth_image = cv2.imread(
            str(self.base_path / frame_json['depth_image']), cv2.IMREAD_ANYDEPTH)
        rgb_image = None
        if 'rgb_image' in frame_json:
            rgb_image = cv2.imread(
                str(self.base_path / frame_json['rgb_image']))
        fg_mask = None
        if 'fg_mask' in frame_json:
            fg_mask = cv2.imread(str(self.base_path / frame_json['fg_mask']))

        info = FrameInfo.from_json(frame_json['info'])
        frame = Frame(info, depth_image, rgb_image=rgb_image, fg_mask=fg_mask)

        return frame

    def __len__(self):
        return len(self.frames_json)


def load_ftb(base_path, ground_truth_model_path=None):
    base_path = Path(base_path)
    with open(str(base_path / "frames.json"), 'r') as stream:
        frame_infos = json.load(stream)['root']

    return FTBDataset(frame_infos, base_path, ground_truth_model_path)
