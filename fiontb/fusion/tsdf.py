import torch

from fiontb.camera import Homogeneous
from fiontb.fiontblib import (fuse_dense_volume, fuse_sparse_volume,
                              DenseVolume)


class TSDFFusion:
    def __init__(self, volume):
        self.volume = volume

    def fuse(self, frame_pcl, kcam, rt_cam):
        height, width = frame_pcl.depth_image.shape
        wpos_image = Homogeneous(rt_cam.cam_to_world) @ frame_pcl.points        
        wpos_image = torch.from_numpy(wpos_image.reshape(height, width, 3))
        cam_center = rt_cam.center()

        # import ipdb; ipdb.set_trace()
        if isinstance(self.volume, DenseVolume):
            fuse_dense_volume(self.volume, wpos_image, cam_center,
                              rt_cam.world_to_cam, kcam.matrix,
                              0.03)
        else:
            fuse_sparse_volume(self.volume, wpos_image, cam_center,
                               rt_cam.world_to_cam, kcam.matrix, 0.03)
