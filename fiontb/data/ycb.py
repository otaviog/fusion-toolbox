"""YCB dataset loading"""

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import h5py
import cv2
from natsort import natsorted

from .datatype import Snapshot
from fiontb.camera import KCamera, RTCamera


def _im2col(im, psize):
    # pylint: disable=invalid-name
    n_channels = 1 if len(im.shape) == 2 else im.shape[0]
    (n_channels, rows, cols) = (1,) * (3 - len(im.shape)) + im.shape

    im_pad = np.zeros((n_channels,
                       int(math.ceil(1.0 * rows / psize) * psize),
                       int(math.ceil(1.0 * cols / psize) * psize)))
    im_pad[:, 0:rows, 0:cols] = im

    final = np.zeros((im_pad.shape[1], im_pad.shape[2], n_channels,
                      psize, psize))
    for c in range(n_channels):
        for x in range(psize):
            for y in range(psize):
                im_shift = np.vstack(
                    (im_pad[c, x:], im_pad[c, :x]))
                im_shift = np.column_stack(
                    (im_shift[:, y:], im_shift[:, :y]))
                final[x::psize, y::psize, c] = np.swapaxes(
                    im_shift.reshape(int(im_pad.shape[1] / psize), psize,
                                     int(im_pad.shape[2] / psize), psize), 1, 2)

    return np.squeeze(final[0:rows - psize + 1, 0:cols - psize + 1])


def _filter_discontinuities(depth_img):
    filt_size = 7
    thresh = 1000

    # Ensure that filter sizes are okay
    assert filt_size % 2 == 1, "Can only use odd filter sizes."

    # Compute discontinuities
    offset = int((filt_size - 1) / 2)
    patches = 1.0 * _im2col(depth_img, filt_size)
    mids = patches[:, :, offset, offset]
    mins = np.min(patches, axis=(2, 3))
    maxes = np.max(patches, axis=(2, 3))

    discont = np.maximum(np.abs(mins - mids),
                         np.abs(maxes - mids))
    mark = discont > thresh

    # Account for offsets
    final_mark = np.zeros((480, 640), dtype=np.uint16)
    final_mark[offset:offset + mark.shape[0],
               offset:offset + mark.shape[1]] = mark

    return depth_img * (1 - final_mark)


def _register_depth_map_authors(unregisteredDepthMap,
                                rgbImage,
                                depthK,
                                rgbK,
                                H_RGBFromDepth):
    # pylint: disable=invalid-name

    unregisteredHeight = unregisteredDepthMap.shape[0]
    unregisteredWidth = unregisteredDepthMap.shape[1]

    registeredHeight = rgbImage.shape[0]
    registeredWidth = rgbImage.shape[1]

    registeredDepthMap = np.zeros((registeredHeight, registeredWidth))

    xyzDepth = np.empty((4, 1))
    xyzRGB = np.empty((4, 1))

    # Ensure that the last value is 1 (homogeneous coordinates)
    xyzDepth[3] = 1

    invDepthFx = 1.0 / depthK[0, 0]
    invDepthFy = 1.0 / depthK[1, 1]
    depthCx = depthK[0, 2]
    depthCy = depthK[1, 2]

    rgbFx = rgbK[0, 0]
    rgbFy = rgbK[1, 1]
    rgbCx = rgbK[0, 2]
    rgbCy = rgbK[1, 2]

    undistorted = np.empty(2)
    for v in range(unregisteredHeight):
        for u in range(unregisteredWidth):

            depth = unregisteredDepthMap[v, u]
            if depth == 0:
                continue

            xyzDepth[0] = ((u - depthCx) * depth) * invDepthFx
            xyzDepth[1] = ((v - depthCy) * depth) * invDepthFy
            xyzDepth[2] = depth

            xyzRGB[0] = (H_RGBFromDepth[0, 0] * xyzDepth[0] +
                         H_RGBFromDepth[0, 1] * xyzDepth[1] +
                         H_RGBFromDepth[0, 2] * xyzDepth[2] +
                         H_RGBFromDepth[0, 3])
            xyzRGB[1] = (H_RGBFromDepth[1, 0] * xyzDepth[0] +
                         H_RGBFromDepth[1, 1] * xyzDepth[1] +
                         H_RGBFromDepth[1, 2] * xyzDepth[2] +
                         H_RGBFromDepth[1, 3])
            xyzRGB[2] = (H_RGBFromDepth[2, 0] * xyzDepth[0] +
                         H_RGBFromDepth[2, 1] * xyzDepth[1] +
                         H_RGBFromDepth[2, 2] * xyzDepth[2] +
                         H_RGBFromDepth[2, 3])

            invRGB_Z = 1.0 / xyzRGB[2]
            undistorted[0] = (rgbFx * xyzRGB[0]) * invRGB_Z + rgbCx
            undistorted[1] = (rgbFy * xyzRGB[1]) * invRGB_Z + rgbCy

            uRGB = int(undistorted[0] + 0.5)
            vRGB = int(undistorted[1] + 0.5)

            if (uRGB < 0 or uRGB >= registeredWidth) or (vRGB < 0 or vRGB >= registeredHeight):
                continue

            registeredDepth = xyzRGB[2]
            if registeredDepth > registeredDepthMap[vRGB, uRGB]:
                registeredDepthMap[vRGB, uRGB] = registeredDepth

    return registeredDepthMap


def _register_rgb_to_depth(depth_img, depth_kcam, rgb_img, rgb_kcam, extrinsic_depth_to_rgb):
    xs, ys = np.meshgrid(np.arange(depth_img.shape[1]),
                         np.arange(depth_img.shape[0]))
    # From image to depth cam space
    points = np.dstack([xs, ys, depth_img]).astype(np.float32)
    points = points.reshape(-1, 3, 1)
    points = depth_kcam.backproject(points)

    # To RGB cam space
    points = np.insert(points, 3, 1.0, axis=1)
    points = np.matmul(extrinsic_depth_to_rgb, points)

    # To RGB image space
    depths = points[:, 2, 0]
    points = rgb_kcam.project(points[:, 0:3])
    points = np.round(points).astype(np.float32)

    map_x = points[:, 0, 0].reshape(depth_img.shape)
    map_y = points[:, 1, 0].reshape(depth_img.shape)

    return map_x, map_y, map_y

    reg_img = np.zeros((depth_img.shape[0], depth_img.shape[1], rgb_img.shape[2]),
                       dtype=rgb_img.dtype)
    reg_height, reg_width = rgb_img.shape[0:2]
    X = np.dstack([xs, ys]).reshape(-1, 2, 1)
    for (x, y), (u, v, _) in zip(X, points.astype(np.int16)):
        if v < 0 or v >= reg_height:
            continue
        if u < 0 or u >= reg_width:
            continue
        reg_img[y, x, :] = rgb_img[v, u, :]

    return map_x, map_y, reg_img


class YCB:
    """YCB indexed snapshots dataset.

    Use `__getitem__` to return a :obj:`fiontb.data.Snapshot`.
    """

    def __init__(self, entries, depth_k_cams, rgb_k_cams, depth_scales, depth_bias,
                 filter_depth=False):
        self.entries = entries
        self.depth_k_cams = depth_k_cams
        self.rgb_k_cams = rgb_k_cams
        self.depth_scales = depth_scales
        self.depth_bias = depth_bias
        self.filter_depth = filter_depth

    def __getitem__(self, idx):
        viewport, depth_file, rgb_file, mask_file, h_rgb_from_depth, rtcam = self.entries[
            idx]

        depth_k_cam = KCamera(self.depth_k_cams[viewport])
        rgb_k_cam = KCamera(self.rgb_k_cams[viewport])

        with h5py.File(depth_file) as hfile:
            depth_img = np.array(hfile["depth"])

        rgb_img = cv2.imread(rgb_file)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        if self.filter_depth:
            depth_img = _filter_discontinuities(depth_img)

        rmap_x, rmap_y, img = _register_rgb_to_depth(depth_img, depth_k_cam,
                                                     rgb_img, rgb_k_cam,
                                                     h_rgb_from_depth)

        rgb_img = cv2.remap(rgb_img, rmap_x, rmap_y, cv2.INTER_LINEAR)
        # Just resizing has the same effect of the registration...
        # rgb_img = cv2.resize(
        #    rgb_img, (depth_img.shape[1], depth_img.shape[0]), cv2.INTER_LINEAR)

        mask_img = cv2.imread(mask_file)
        if mask_img is not None:
            mask_img = cv2.remap(mask_img, rmap_x, rmap_y, cv2.INTER_NEAREST)
        #    mask_img = cv2.resize(
        #        mask_img, (rgb_img.shape[1], rgb_img.shape[0]), cv2.INTER_NEAREST)
            mask_img = mask_img[:, :, 0] == 0

        snap = Snapshot(
            depth_img, depth_k_cam,
            # depth_scale=self.depth_scales[viewport],
            depth_scale=0.001,
            depth_bias=self.depth_bias[viewport],
            depth_max=np.iinfo(np.uint16).max,
            rgb_image=rgb_img,
            rgb_kcam=rgb_k_cam,
            fg_mask=mask_img,
            rt_cam=RTCamera(rtcam))

        return snap

    def __len__(self):
        return len(self.entries)


def _get_rgb_from_depth(calibration, camera, reference_camera):
    ir_key = "H_{0}_ir_from_{1}".format(camera, reference_camera)
    rgb_key = "H_{0}_from_{1}".format(camera, reference_camera)

    rgb_from_ref = calibration[rgb_key][:]
    ir_from_ref = calibration[ir_key][:]

    return np.matmul(rgb_from_ref, np.linalg.inv(ir_from_ref))
    # return np.matmul(ir_from_ref, np.linalg.inv(rgb_from_ref))


def _get_pose(hfile, rgb_from_ref):
    ref_table_pose = hfile['H_table_from_reference_camera']
    cam = np.matmul(rgb_from_ref, np.linalg.inv(ref_table_pose))
    # cam = np.matmul(rgb_from_ref, ref_table_pose)

    # cam = np.linalg.inv(cam) 
    return cam

def load_ycb_object(base_path):
    base_path = Path(base_path)

    viewports = ["NP1", "NP2", "NP3", "NP4", "NP5"]

    depth_k_cams = defaultdict(list)
    rgb_k_cams = defaultdict(list)
    h_rgb_from_depth = defaultdict(list)
    depth_scales = {}
    depth_bias = {}

    with h5py.File(base_path / "calibration.h5") as hfile:
        for vp in viewports:
            depth_k_cams[vp] = np.array(
                hfile["{}_depth_K".format(vp)])  # was ir_K
            rgb_k_cams[vp] = np.array(hfile["{}_rgb_K".format(vp)])
            h_rgb_from_depth[vp] = _get_rgb_from_depth(hfile, vp, 'NP5')
            depth_scales[vp] = float(
                hfile['{}_ir_depth_scale'.format(vp)][:])
            depth_bias[vp] = float(hfile['{}_depth_bias'.format(vp)][:])

    entries = []
    for viewport in viewports:
        view_images = base_path.glob("{}_*.jpg".format(viewport))
        view_images = natsorted(view_images,
                                key=lambda key: str(key))

        for rgb_filepath in view_images:
            depth_filepath = rgb_filepath.with_suffix(".h5")

            if not depth_filepath.exists():
                raise RuntimeError("Missing '{}' file".format(depth_filepath))
            mask_filepath = base_path / "masks" / \
                (rgb_filepath.stem + '_mask.pbm')

            img_id = int(rgb_filepath.stem.split('_')[1])
            with h5py.File(base_path / 'poses' / 'NP5_{}_pose.h5'.format(img_id)) as hfile:
                rtcam = _get_pose(hfile, h_rgb_from_depth[viewport])

            entries.append((viewport, str(depth_filepath), str(rgb_filepath),
                            str(mask_filepath),
                            h_rgb_from_depth[viewport],
                            rtcam))

    return YCB(entries, depth_k_cams, rgb_k_cams, depth_scales,
               depth_bias, filter_depth=True)


def _main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('ycb_base_dir')

    args = parser.parse_args()

    ycb_base_dir = Path(args.ycb_base_dir)
    objects = ycb_base_dir.glob("*")
    objects = [obj for obj in objects
               if obj.is_dir()]

    max_depth = 0

    for obj in tqdm(objects):
        ycb_ds = load_ycb_object(obj)
        for entry in ycb_ds.entries:
            depth_file = entry[1]

            with h5py.File(depth_file) as hfile:
                depth_img = hfile['depth'][:]
                max_depth = max(max_depth, depth_img.max())

    print("Maximum YCB depth value: {}".format(max_depth))


if __name__ == '__main__':
    _main()
