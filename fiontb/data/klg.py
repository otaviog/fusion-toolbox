import struct

import numpy as np
import cv2
import zlib

from .datatype import Snapshot


def _read_frame_header(stream):
    timestamp = struct.unpack('q', stream.read(8))[0]
    depth_size = struct.unpack('i', stream.read(4))[0]
    rgb_size = struct.unpack('i', stream.read(4))[0]

    return timestamp, depth_size, rgb_size


def _read_frame_pointers(stream, num_frames):
    frame_ptrs = []
    for _ in range(num_frames):
        frame_ptrs.append(stream.tell())
        _, depth_size, rgb_size = _read_frame_header(stream)
        stream.seek(depth_size + rgb_size, 1)
    return frame_ptrs


class KLG:

    def __init__(self, filepath):
        with open(filepath, 'rb') as stream:
            num_frames = struct.unpack('i', stream.read(4))[0]
            self.frame_ptrs = _read_frame_pointers(stream, num_frames)

        self.stream = open(filepath, 'rb')

    def __getitem__(self, idx):
        seek = self.frame_ptrs[idx]
        self.stream.seek(seek, 0)

        _, depth_size, rgb_size = _read_frame_header(self.stream)
        raw_depth = self.stream.read(depth_size)
        jpg_rgb = self.stream.read(rgb_size)

        rgb_img = cv2.imdecode(np.fromstring(jpg_rgb, dtype=np.uint8), 1)

        depth_img = np.fromstring(zlib.decompress(raw_depth), dtype=np.uint16)
        depth_img = depth_img.reshape(rgb_img.shape[0:2])

        import ipdb; ipdb.set_trace()

        snap = Snapshot(depth_image=depth_img, rgb_image=rgb_img)

        return snap

    def __len__(self):
        return len(self.frame_ptrs)
