"""Custom RGB-D SLAM dataset format developed by SLAM Toolbox.
"""
from pathlib import Path
import json

import cv2
from tqdm import tqdm
import torch
import numpy as np

from slamtb.frame import Frame, FrameInfo
from slamtb.camera import RTCamera


class FTBDataset:
    """A frame dataset.
    """

    def __init__(self, frames_json, frame_infos, base_path, ground_truth_model_path=None):
        self.frames_json = frames_json
        self.frame_infos = frame_infos
        self.base_path = base_path
        self.ground_truth_model_path = ground_truth_model_path

    def get_info(self, idx):
        """Get the frame information at an index:

        Args:

            idx (int): Frame index.

        Returns: (`FrameInfo`): Frame information.

        """

        return self.frame_infos[idx]

    def set_info(self, idx, info):
        """
        Set the information for a frame.

        Args:

            idx (int): Frame index.

            info (`FrameInfo`): Frame information.
        """

        self.frame_infos[idx] = info

    def __getitem__(self, idx):
        frame_json = self.frames_json[idx]

        depth_image = cv2.imread(
            str(self.base_path / frame_json['depth_image']),
            cv2.IMREAD_ANYDEPTH).astype(np.int32)
        rgb_image = None
        if 'rgb_image' in frame_json:
            rgb_image = cv2.imread(
                str(self.base_path / frame_json['rgb_image']))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        seg_image = None
        if 'segmentation' in frame_json:
            seg_image = cv2.imread(
                str(self.base_path / frame_json['segmentation']), cv2.IMREAD_ANYDEPTH)
            seg_image = seg_image.astype(np.int16)
        info = self.frame_infos[idx]
        frame = Frame(info, depth_image,
                      rgb_image=rgb_image, seg_image=seg_image)

        return frame

    def __len__(self):
        return len(self.frames_json)


def load_ftb(base_path, info_file=None):
    r"""Load a scene dataset.

    Args:

        base_path (str): Path to the base directory.

        info_file (str, optional): Path to the frame informations json
         file. If not specified, then it will look for the
         <base_path>/frames.json file.

    Returns: FTBDataset: a instanced dataset object.
    """

    base_path = Path(base_path)
    if info_file is None:
        info_file = str(base_path / "frames.json")

    with open(info_file, 'r') as stream:
        frames = json.load(stream)['root']

    frame_infos = [FrameInfo.from_json(frame_json['info'])
                   for frame_json in frames]

    return FTBDataset(frames, frame_infos, base_path)


def load_trajectory(json_file):
    """Reads the trajectory from a FTB frame information file.

    Args:

        json_file (str): Path to the json frame information file.

    Returns: (Dict[RTCamera]): Camera trajectory, keys are timestamps
     and values are cameras.

    """

    with open(str(json_file), 'r') as file:
        json_dict = json.load(file)

    trajectory = {traj['timestamp']:
                  RTCamera(torch.tensor(traj['rt_cam'])) for traj in json_dict}

    return trajectory


def write_ftb(base_path, dataset, prefix="frame_", max_frames=None, start_frame=0):
    """Write a dataset using our FTB format.

    Args:

        base_path (str): Output directory path.

        dataset (object): Any slamtb dataset object.

        prefix (str): Image files prefix.

        max_frames (int, optional): Maximum number to frames to write.

        start_frame (int, optional): Begining frame to start writing.

    """

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    frames = []

    if max_frames is not None:
        max_frames = min(start_frame+max_frames, len(dataset))
    else:
        max_frames = len(dataset)

    for i in tqdm(range(start_frame, max_frames)):
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
                        cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2BGR))
            frame_dict['rgb_image'] = rgb_file

        if frame.seg_image is not None:
            seg_file = "{}{:05d}_seg.png".format(prefix, i)
            cv2.imwrite(str(base_path / seg_file),
                        frame.seg_image.astype(np.uint16))
            frame_dict['segmentation'] = seg_file

        frames.append(frame_dict)

    with open(str(base_path / "frames.json"), 'w') as stream:
        json.dump({'root': frames}, stream, indent=1)
