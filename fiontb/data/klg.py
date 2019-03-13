"""KLG reader as an indexed snapshot dataset
"""
import struct
import zlib

import numpy as np
import cv2
from tqdm import tqdm

from fiontb.camera import KCamera

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
    """
    """

    def __init__(self, filepath):
        with open(filepath, 'rb') as stream:
            num_frames = struct.unpack('i', stream.read(4))[0]
            self.frame_ptrs = _read_frame_pointers(stream, num_frames)

        self.stream = open(filepath, 'rb')
        self.kcam = KCamera(np.eye(3))
        self.trajectory = None
        self.depth_scale = 1.0

    def __del__(self):
        self.stream.close()

    def __getitem__(self, idx):
        seek = self.frame_ptrs[idx]
        self.stream.seek(seek, 0)

        _, depth_size, rgb_size = _read_frame_header(self.stream)
        raw_depth = self.stream.read(depth_size)
        jpg_rgb = self.stream.read(rgb_size)

        rgb_img = cv2.imdecode(np.frombuffer(jpg_rgb, dtype=np.uint8), 1)

        depth_img = np.frombuffer(zlib.decompress(raw_depth), dtype=np.uint16)

        depth_img = depth_img.reshape(rgb_img.shape[0:2])

        rt_cam = None
        if self.trajectory is not None:
            if idx < len(self.trajectory):
                rt_cam = self.trajectory[idx]
            else:
                rt_cam = self.trajectory[-1]

        snap = Snapshot(depth_image=depth_img, rgb_image=rgb_img,
                        kcam=self.kcam,
                        rt_cam=rt_cam,
                        depth_scale=self.depth_scale)

        return snap

    def __len__(self):
        return len(self.frame_ptrs)


def write_klg(dataset, stream, format_depth_scale, max_frames=None):
    """Writes a :obj:`fiontb.Snapshot` dataset to .klg file format.

    Args:

        dataset (list[:obj:`fiontb.Snapshot`]): A dataset of RGB-D
         snapshots.

        stream (file): binary output stream.

        format_depth_scale (float):

        max_frames (int, optional): maximum number of frames to write,
         default is to use dataset length.
    """

    if max_frames is not None:
        max_frames = min(max_frames, len(dataset))
    else:
        max_frames = len(dataset)

    stream.write(struct.pack('i', max_frames))

    rt_cams = []
    for i in tqdm(range(max_frames), desc="Writing KLG file"):
        snap = dataset[i]

        depth_image = snap.depth_image
        depth_image = depth_image*snap.depth_scale
        depth_image = depth_image*format_depth_scale

        depth_image = depth_image.astype(np.uint16)
        compress_depth = zlib.compress(depth_image)
        _, jpg_rgb = cv2.imencode('.jpg', snap.rgb_image)

        if snap.timestamp is not None:
            if isinstance(snap.timestamp, float):
                timestamp = int(snap.timestamp*10000000)
            else:
                timestamp = snap.timestamp
        else:
            timestamp = i + 1

        stream.write(struct.pack('q', timestamp))
        stream.write(struct.pack('i', len(compress_depth)))
        stream.write(struct.pack('i', len(jpg_rgb)))

        stream.write(compress_depth)
        stream.write(jpg_rgb)

        if snap.rt_cam is not None:
            rt_cam = snap.rt_cam

            rt_cams.append((timestamp, rt_cam))

    return rt_cams
