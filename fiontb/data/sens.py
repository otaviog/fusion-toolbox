"""Sens reader and writer module.
"""

import struct
import zlib

import cv2
import numpy as np
from tqdm import tqdm

from fiontb.camera import RTCamera, KCamera

from .datatype import Snapshot

COLOR_COMPRESSION_INT_TO_STR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COLOR_COMPRESSION_STR_TO_INT = {v: k for k,
                                v in COLOR_COMPRESSION_INT_TO_STR.items()}
DEPTH_COMPRESSION_INT_TO_STR = {-1: 'unknown',
                                0: 'raw_ushort', 1: 'zlib_ushort',
                                # 2: 'occi_ushort'
                                }
DEPTH_COMPRESSION_STR_TO_INT = {v: k for k,
                                v in DEPTH_COMPRESSION_INT_TO_STR.items()}


def write_sens(filepath, dataset, snap0, sensor_name="fiontb"):

    with open(str(filepath), 'wb') as file:
        file.write(struct.pack('I', 4))
        file.write(struct.pack('Q', len(sensor_name)))

        file.write(struct.pack('{0}s'.format(
            len(sensor_name)), sensor_name.encode()))

        color_intrinsic = np.eye(4)
        file.write(struct.pack('16f', *color_intrinsic.flatten().tolist()))

        color_extrinsic = np.eye(4)
        file.write(struct.pack('16f', *color_extrinsic.flatten().tolist()))

        depth_intrinsic = np.eye(4)
        depth_intrinsic[:3, :3] = snap0.kcam.matrix
        file.write(struct.pack('16f', *depth_intrinsic.flatten().tolist()))

        depth_extrinsic = np.eye(4)
        file.write(struct.pack('16f', *depth_extrinsic.flatten().tolist()))

        file.write(struct.pack('i', COLOR_COMPRESSION_STR_TO_INT['png']))
        file.write(struct.pack(
            'i', DEPTH_COMPRESSION_STR_TO_INT['zlib_ushort']))

        file.write(struct.pack('I', snap0.rgb_image.shape[1]))
        file.write(struct.pack('I', snap0.rgb_image.shape[0]))

        file.write(struct.pack('I', snap0.depth_image.shape[1]))
        file.write(struct.pack('I', snap0.depth_image.shape[0]))

        file.write(struct.pack('f', 1.0))

        file.write(struct.pack('Q', len(dataset)))

        timestamp = 0
        rt_cams = []
        for i in tqdm(range(len(dataset)), total=len(dataset), desc="Writing {}".format(filepath)):
            snap = dataset[i]

            file.write(struct.pack(
                '16f', *snap.rt_cam.matrix.flatten().tolist()))

            if snap.timestamp is not None:
                if isinstance(snap.timestamp, float):
                    timestamp = int(snap.timestamp*10000000)
                else:
                    timestamp = snap.timestamp
            else:
                timestamp = i + 1

            file.write(struct.pack('Q', timestamp))
            file.write(struct.pack('Q', timestamp))

            _, rgb_data = cv2.imencode('.png', snap.rgb_image)
            depth_data = zlib.compress(snap.depth_image)

            file.write(struct.pack('Q', len(rgb_data)))
            file.write(struct.pack('Q', len(depth_data)))

            file.write(rgb_data)
            file.write(depth_data)

            if snap.rt_cam is not None:
                rt_cams.append((snap.timestamp, snap.rt_cam))

    return rt_cams


class _FrameInfo:
    def __init__(self, rt_cam, color_ts, depth_ts,
                 color_offset, color_size,
                 depth_offset, depth_size):
        self.rt_cam = rt_cam
        self.color_ts = color_ts
        self.depth_ts = depth_ts

        self.color_offset = color_offset
        self.color_size = color_size

        self.depth_offset = depth_offset
        self.depth_size = depth_size


class SensDataset:
    def _read_header(self):
        # pylint: disable=unused-variable

        version = struct.unpack('I', self.file.read(4))[0]

        strlen = struct.unpack('Q', self.file.read(8))[0]
        sensor_name = ''.join(struct.unpack(
            'c'*strlen, self.file.read(strlen))[0].decode())

        self.intrinsic_color = np.asarray(struct.unpack(
            'f'*16, self.file.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.extrinsic_color = np.asarray(struct.unpack(
            'f'*16, self.file.read(16*4)), dtype=np.float32).reshape(4, 4)

        intrinsic_depth = np.asarray(struct.unpack(
            'f'*16, self.file.read(16*4)), dtype=np.float32).reshape(4, 4)
        self.kcam = KCamera(intrinsic_depth[:3, :3])
        self.extrinsic_depth = np.asarray(struct.unpack(
            'f'*16, self.file.read(16*4)), dtype=np.float32).reshape(4, 4)

        self.color_compression_type = COLOR_COMPRESSION_INT_TO_STR[
            struct.unpack('i', self.file.read(4))[0]]
        try:
            depth_compression = struct.unpack('i', self.file.read(4))[0]
            self.depth_compression_type = DEPTH_COMPRESSION_INT_TO_STR[depth_compression]
        except KeyError:
            raise RuntimeError(
                "Unknown compression id {}".format(depth_compression))

        self.color_width = struct.unpack('I', self.file.read(4))[0]
        self.color_height = struct.unpack('I', self.file.read(4))[0]
        self.depth_width = struct.unpack('I', self.file.read(4))[0]
        self.depth_height = struct.unpack('I', self.file.read(4))[0]
        self.depth_shift = struct.unpack('f', self.file.read(4))[0]

        self.num_frames = struct.unpack('Q', self.file.read(8))[0]

    def __init__(self, filepath):
        self.file = open(filepath, 'rb')
        self._read_header()

        self.frame_infos = []

        for i in range(self.num_frames):
            camera_to_world = np.asarray(
                struct.unpack('16f', self.file.read(16*4)),
                dtype=np.float32).reshape(4, 4)
            color_ts = struct.unpack('Q', self.file.read(8))[0]
            depth_ts = struct.unpack('Q', self.file.read(8))[0]

            color_size = struct.unpack('Q', self.file.read(8))[0]
            depth_size = struct.unpack('Q', self.file.read(8))[0]

            color_offset = self.file.tell()
            depth_offset = color_offset + color_size
            self.file.seek(color_size + depth_size, 1)

            self.frame_infos.append(
                _FrameInfo(RTCamera(camera_to_world), color_ts, depth_ts,
                           color_offset, color_size,
                           depth_offset, depth_size))
        import ipdb
        ipdb.set_trace()
        pass

    def __del__(self):
        self.file.close()

    def __getitem__(self, idx):
        frame_info = self.frame_infos[idx]

        self.file.seek(frame_info.color_offset, 0)

        color_data = self.file.read(frame_info.color_size)
        if self.color_compression_type in ('png', 'jpeg'):
            color_img = cv2.imdecode(
                np.frombuffer(color_data, dtype=np.uint8), 1)
        elif self.color_compression_type == 'raw':
            color_img = np.frombuffer(color_data, dtype=np.uint8)
            color_img = color_img.reshape(
                self.color_height, self.color_width, 3)

        depth_data = self.file.read(frame_info.depth_size)

        if self.depth_compression_type == 'zlib_ushort':
            depth_data = zlib.decompress(depth_data)

        depth_img = np.frombuffer(depth_data, dtype=np.uint16)
        depth_img = depth_img.reshape(self.depth_height, self.depth_width)

        snap = Snapshot(depth_image=depth_img, rgb_image=color_img,
                        kcam=self.kcam, rt_cam=frame_info.rt_cam,
                        depth_scale=1.0/self.depth_shift)

        return snap

    def __len__(self):
        return self.num_frames


def load_sens(filepath):
    return SensDataset(filepath)
