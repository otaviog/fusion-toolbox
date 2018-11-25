"""InteriorNet parsing
"""

from pathlib import Path
from collections import namedtuple

import numpy as np
from quaternion import quaternion, as_rotation_matrix
import cv2

from fusionkit.camera import Camera, IntrinsicCamera, ExtrinsicCamera
from .datatype import Snapshot

Entry = namedtuple("InteriorNetEntry", ["cam", "depth_path", "rgb_path"])


class InteriorNet:

    def __init__(self, trajectory):
        self.trajectory = trajectory

    def __getitem__(self, idx):
        return InteriorNet.load_snapshot(self.trajectory[idx])

    def __len__(self):
        return len(self.trajectory)

    @staticmethod
    def load_snapshot(innet_entry):
        cimg = cv2.imread(str(innet_entry.rgb_path))
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
        cimg = np.flipud(cimg)
        cimg = cimg/255

        dimg = cv2.imread(str(innet_entry.depth_path))

        dimg = dimg[:, :, 0]
        dimg = np.flipud(dimg)
        max_depth = 1  # dimg.max()
        # dimg = (dimg.max() - dimg)/max_depth
        dimg = dimg/max_depth

        return Snapshot(depth_image=dimg, color_image=cimg,
                        intr_cam=innet_entry.cam.intrinsic_cam,
                        extr_cam=innet_entry.cam.extrinsic_cam)


def load_interiornet(base_path):
    """Parses an InteriorNet camera trajectory.

    Args:

        base_path (str or :obj:`Path`): base path a camera's
         trajectory.

    """
    base_path = Path(base_path)

    def glob_img_list(imgs_path):
        img_list = imgs_path.glob("*.png")
        return sorted(img_list,
                      key=lambda img_path: int(img_path.stem))
    rgb_img_list = glob_img_list(base_path / 'rgb')
    depth_img_list = glob_img_list(base_path / 'depth')

    with open(base_path / "cam0.info") as file:
        file.readline()  # first comment
        img_width, img_height = map(float, file.readline().split())
        file.readline()  # comment
        focal_x, focal_y = map(float, file.readline().split())
        file.readline()  # comment
        center_x, center_y = map(float, file.readline().split())
        file.readline()  # comment
        undist_coeff = map(float, file.readline().split())

    intr_cam = IntrinsicCamera.create_from_params(
        focal_x, focal_y, (center_x, center_y), undist_coeff,
        image_size=(img_width, img_height))

    camera_list = []
    with open(base_path / "cam0_gt.visim") as file:
        for line in file:
            if line.startswith("#"):
                continue
            entry = [float(v) for v in line.split(',')]
            pos = np.array([entry[1], entry[2], entry[3]])
            quat = quaternion(entry[4], entry[5], entry[6], entry[7])

            ext_cam = ExtrinsicCamera.create_from_params(
                position=pos,
                rotation_matrix=as_rotation_matrix(quat)
                #rotation_matrix=np.eye(3)
            )
            camera = Camera(intr_cam, ext_cam)
            camera_list.append(camera)

    trajectory = [Entry(camera, depth_img, rgb_img)
                  for camera, depth_img, rgb_img
                  in zip(camera_list, depth_img_list, rgb_img_list)]
    return InteriorNet(trajectory)
