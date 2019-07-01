import rflow


class LoadMesh(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from tenviz.io import read_3dobject

        return read_3dobject(resource.filepath).torch()


class MeshToPCL(rflow.Interface):
    def evaluate(self, mesh_geo):
        from fiontb.pointcloud import PointCloud
        return PointCloud(mesh_geo.verts,
                          mesh_geo.colors,
                          mesh_geo.normals)
