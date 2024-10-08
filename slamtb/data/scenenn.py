"""Parser for the SceneNN RGB-D dataset from .oni files.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
import natsort

import onireader


from slamtb.camera import KCamera, RTCamera
from slamtb.frame import Frame, FrameInfo


KINECT2_KCAM = KCamera(torch.tensor([[356.769928, 0.0, 251.563446],
                                     [0.0, 430.816498, 237.563446],
                                     [0.0, 0.0, 1.0]], dtype=torch.float))

ASUS_KCAM = KCamera(torch.tensor([[544.47329, 0.0, 320],
                                  [0.0, 544.47329, 240],
                                  [0.0, 0.0, 1.0]], dtype=torch.float))


class SceneNN:
    """(Almost)-Indexed snapshot dataset for SceneNN. Note due to oni
    playback, this class always advanced one frame no matter which
    indexed is passed. Use :func:`rewind` to go to the begining.

    """

    def __init__(self, oni_filepath, trajectory, kcam, mask_file_list=None):
        self._oni_filepath = oni_filepath
        self.rewind()

        self.trajectory = trajectory
        self.kcam = kcam

        self.first_frame_id = None
        self.last_idx = None
        self.cache = None

        self.mask_file_list = mask_file_list
        self._debug = False

    def rewind(self):
        """Rewinds the data to the begining frames.
        """
        # self.ni_dev.seek doesn't work
        self.ni_dev = None
        self.ni_dev = onireader.Device()
        self.ni_dev.open(str(self._oni_filepath))
        self.ni_dev.start()

    def _getnext_pair(self):
        depth_img, depth_ts, _ = self.ni_dev.read_depth()
        rgb_img, rgb_ts, _ = self.ni_dev.read_color()

        k_time_diff = 33000
        diff = abs(rgb_ts - depth_ts)

        while diff > k_time_diff:
            if rgb_ts > depth_ts:
                depth_img, depth_ts, _ = self.ni_dev.read_depth()
            else:
                rgb_img, rgb_ts, _ = self.ni_dev.read_color()

            diff = abs(rgb_ts - depth_ts)
            if self._debug:
                print("Skiping rgb {} and depth {}".format(rgb_ts, depth_ts))
        return depth_img.astype(np.int32), rgb_img, depth_ts

    def __getitem__(self, idx):
        # pylint: disable=unused-variable
        if self.last_idx != idx:
            self.cache = self._getnext_pair()
            self.last_idx = idx

        depth_img, rgb_img, depth_ts = self.cache

        rt_mtx = self.trajectory[idx]

        info = FrameInfo(self.kcam, depth_scale=0.001,
                         timestamp=depth_ts, rt_cam=RTCamera(rt_mtx))
        seg_image = None
        if self.mask_file_list is not None:
            filename = self.mask_file_list[idx]

            seg_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # https://github.com/hkust-vgd/shrec17/blob/master/mask_from_label/mask_from_label.cpp

            seg_image = seg_image.astype(np.uint32)
            temp = np.bitwise_or(np.left_shift(seg_image[:, :, 0], 24),
                                 np.left_shift(seg_image[:, :, 1], 16))
            temp = np.bitwise_or(temp, np.left_shift(seg_image[:, :, 2], 8))
            seg_image = np.bitwise_or(
                temp, seg_image[:, :, 3]).astype(np.int32)

        return Frame(info, depth_img, rgb_image=rgb_img, seg_image=seg_image)

    def __len__(self):
        return len(self.trajectory)

    def get_info(self, idx):
        rt_mtx = self.trajectory[idx]

        info = FrameInfo(self.kcam, depth_scale=0.001, rt_cam=RTCamera(rt_mtx))
        return info


def load_annotations(xml_filepath):
    root = ET.parse(xml_filepath).getroot()
    annotations = []
    for child in root:
        annotations.append(child.attrib)

    return annotations


def load_scenenn(oni_filepath, traj_filepath, k_cam_dev='asus',
                 mask_dirpath=None):
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
            # cam space to world space
            rt_mtx = torch.tensor(curr_entry, dtype=torch.float)

            assert rt_mtx.shape == (4, 4)
            trajectory.append(rt_mtx)

    k_cams = {'asus': ASUS_KCAM, 'kinect2': KINECT2_KCAM}

    if k_cam_dev not in k_cams:
        raise RuntimeError("Undefined {} camera intrinsics. Use: {}".format(
            k_cam_dev, k_cam_dev.keys()))

    mask_file_list = None
    if mask_dirpath is not None:
        mask_file_list = natsort.natsorted(
            [str(filepath) for filepath in Path(mask_dirpath).glob("*.png")])
        if len(mask_file_list) == 0:
            mask_file_list = None
    return SceneNN(oni_filepath, trajectory, k_cams[k_cam_dev],
                   mask_file_list=mask_file_list)
