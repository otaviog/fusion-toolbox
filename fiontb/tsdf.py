"""
"""

import numpy as np


def tsdf():
    pass


class TSDFFusion:
    def __init__(self, vol_shape, voxel_scale):
        self.volume = np.ones(vol_shape)*0
        self.voxel_scale = voxel_scale

    def add_point(self, point, cam_pos):
        x, y, z = point
        depth, height, width = self.volume.shape
        col = int((x*self.voxel_scale + width/2))
        row = int((y*self.voxel_scale + height/2))
        chn = int((z*self.voxel_scale + depth/2))

        dst = np.linalg.norm(cam_pos - point)
        self.volume[chn, row, col] = dst

    def update(self, depth_image, kcam, rtcam):
        vdepth, vheight, vwidth = self.volume.shape

        xs = np.arange(0, vwidth)
        ys = np.arange(0, vheight)
        zs = np.arange(0, vdepth)

        vol_pos = np.dstack([grid.flatten()
                             for grid in np.meshgrid(xs, ys, zs)]).squeeze()

        world_pos = vol_pos - (vwidth*0.5, vheight*0.5, vdepth*0.5)
        world_pos = world_pos*self.voxel_scale

        cam_pos = rtcam.transform_world_to_cam(world_pos)
        cam_z = cam_pos[:, 2, 0]
        valid_z = cam_z > 0.05
        cam_pos = cam_pos[valid_z]
        vol_pos = vol_pos[valid_z]
        
        img_pos = kcam.project_cam_to_image(cam_pos)
        for i, ipos in enumerate(img_pos):
            drow = int(ipos[1])
            dcol = int(ipos[0])

            if (drow < 0 or drow >= depth_image.shape[0] or
                dcol < 0 or dcol >=
                    depth_image.shape[1]):
                continue

            vol_z, vol_y, vol_x = vol_pos[i]
            dev_z = depth_image[drow, dcol]
            img_z = img_pos[i, 2]

            sdf = np.abs(dev_z - img_z) > 0.1

            self.volume[vol_z, vol_y, vol_x] = sdf
