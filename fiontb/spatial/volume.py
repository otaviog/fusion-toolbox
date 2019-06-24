import fiontb._cfiontb as _cfiontb
from fiontb.pointcloud import PointCloud


class DenseVolume(_cfiontb.DenseVolume):
    def __init__(self, resolution, voxel_size, begin_pos):
        super(DenseVolume, self).__init__(resolution, voxel_size, begin_pos)

    def to_point_cloud(self):
        points = super(DenseVolume, self).to_point_cloud()
        return PointCloud(points)
