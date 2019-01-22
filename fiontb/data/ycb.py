"""YCB dataset loading"""

from collections import defaultdict
from pathlib import Path
import math

import numpy as np
import h5py
import cv2

from .datatype import Snapshot
from fiontb.camera import KCamera


def _im2col(im, psize):
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


def _register_depth_map(depth_img, rgb_img, depth_kcam,
                        rgb_kcam, hom_rgb_from_depth):

    depth_height, depth_width = depth_img.shape
    rgb_height, rgb_width = rgb_img.shape[0:2]

    reg_depth = np.zeros((rgb_height, rgb_width))

    inv_depth_fx = 1.0 / depth_kcam[0, 0]
    inv_depth_fy = 1.0 / depth_kcam[1, 1]
    depth_cx = depth_kcam[0, 2]
    depth_cy = depth_kcam[1, 2]

    rgb_fx, rgb_fy = rgb_kcam[0, 0], rgb_kcam[1, 1]
    rgb_cx, rgb_cy = rgb_kcam[0, 2], rgb_kcam[1, 2]

    for v_depth in range(depth_height):
        for u_depth in range(depth_width):

            depth = depth_img[v_depth, u_depth]
            if depth == 0:
                continue

            xyz_depth = np.array([((u_depth - depth_cx) * depth) * inv_depth_fx,
                                  ((v_depth - depth_cy) * depth) * inv_depth_fy,
                                  depth, 1.0])
            xyz_rgb = np.matmul(hom_rgb_from_depth, xyz_depth)
            xyz_rgb[0] = (hom_rgb_from_depth[0, 0] * xyz_depth[0] +
                          hom_rgb_from_depth[0, 1] * xyz_depth[1] +
                          hom_rgb_from_depth[0, 2] * xyz_depth[2] +
                          hom_rgb_from_depth[0, 3])
            xyz_rgb[1] = (hom_rgb_from_depth[1, 0] * xyz_depth[0] +
                          hom_rgb_from_depth[1, 1] * xyz_depth[1] +
                          hom_rgb_from_depth[1, 2] * xyz_depth[2] +
                          hom_rgb_from_depth[1, 3])
            xyz_rgb[2] = (hom_rgb_from_depth[2, 0] * xyz_depth[0] +
                          hom_rgb_from_depth[2, 1] * xyz_depth[1] +
                          hom_rgb_from_depth[2, 2] * xyz_depth[2] +
                          hom_rgb_from_depth[2, 3])

            inv_rgb_z = 1.0 / xyz_rgb[2]

            u_rgb = (rgb_fx * xyz_rgb[0]) * inv_rgb_z + rgb_cx
            u_rgb = int(u_rgb + 0.5)

            v_rgb = (rgb_fy * xyz_rgb[1]) * inv_rgb_z + rgb_cy
            v_rgb = int(v_rgb + 0.5)

            if (0 <= u_rgb < rgb_width) and (0 <= v_rgb < rgb_height):
                reg_depth[v_rgb, u_rgb] = max(
                    xyz_rgb[2], reg_depth[v_rgb, u_rgb])

    return reg_depth


class YCB:
    def __init__(self, entries, depth_k_cams, rgb_k_cams, depth_scales, depth_bias,
                 filter_depth=False):
        self.entries = entries
        self.depth_k_cams = depth_k_cams
        self.rgb_k_cams = rgb_k_cams
        self.depth_scales = depth_scales
        self.depth_bias = depth_bias
        self.filter_depth = filter_depth

    def __getitem__(self, idx):
        viewport, depth_file, rgb_file, mask_file, h_rgb_from_depth = self.entries[idx]

        depth_k_cam = self.depth_k_cams[viewport]
        rgb_k_cam = self.rgb_k_cams[viewport]

        with h5py.File(depth_file) as hfile:
            depth_img = np.array(hfile["depth"])

        depth_img = depth_img

        depth_size = (depth_img.shape[1], depth_img.shape[0])

        rgb_img = cv2.imread(rgb_file)
        rgb_img = cv2.resize(rgb_img, depth_size)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        mask_img = cv2.imread(mask_file)
        mask_img = cv2.resize(mask_img, depth_size,
                              interpolation=cv2.INTER_NEAREST)
        mask_img = mask_img[:, :, 0] == 0

        if self.filter_depth:
            depth_img = _filter_discontinuities(depth_img)

        # depth_img = _register_depth_map(depth_img, rgb_img, depth_k_cam, rgb_k_cam,
        # h_rgb_from_depth)

        return Snapshot(depth_img, KCamera(depth_k_cam),
                        depth_scale=self.depth_scales[viewport],
                        depth_bias=self.depth_bias[viewport],
                        rgb_image=rgb_img,
                        rgb_kcam=KCamera(rgb_k_cam),
                        fg_mask=mask_img)

    def __len__(self):
        return len(self.entries)


def _get_rgb_from_depth(calibration, camera, reference_camera):
    ir_key = "H_{0}_ir_from_{1}".format(camera, reference_camera)
    rgb_key = "H_{0}_from_{1}".format(camera, reference_camera)
    # import ipdb; ipdb.set_trace()
    rgb_from_ref = calibration[rgb_key][:]
    ir_from_ref = calibration[ir_key][:]

    return np.dot(rgb_from_ref, np.linalg.inv(ir_from_ref))


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
            depth_k_cams[vp] = np.array(hfile["{}_ir_K".format(vp)])
            rgb_k_cams[vp] = np.array(hfile["{}_rgb_K".format(vp)])
            h_rgb_from_depth[vp] = _get_rgb_from_depth(hfile, vp, 'NP5')
            depth_scales[vp] = float(hfile['{}_depth_scale'.format(vp)][:])
            depth_bias[vp] = float(hfile['{}_depth_bias'.format(vp)][:])

    entries = []
    for viewport in viewports:
        view_images = sorted(base_path.glob("{}_*.jpg".format(viewport)))

        for rgb_filepath in view_images:
            depth_filepath = rgb_filepath.with_suffix(".h5")
            mask_filepath = base_path / "masks" / \
                (rgb_filepath.stem + '_mask.pbm')
            entries.append((viewport, str(depth_filepath), str(rgb_filepath),
                            str(mask_filepath),
                            h_rgb_from_depth[viewport]
                            ))

    return YCB(entries, depth_k_cams, rgb_k_cams, depth_scales,
               depth_bias)
