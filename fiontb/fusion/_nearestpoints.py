from cProfile import Profile

import torch
from scipy.spatial import cKDTree

from fiontb.camera import Homogeneous
#from fiontb.fiontblib import IndexMap
#from fiontb.sparse.octtree import OctTree
from fiontb.fiontblib import Octree


class KDTree:
    def __init__(self, points, device):
        self.tree = cKDTree(points.cpu(), balanced_tree=False)
        self.device = device

    def query(self, points, max_k, radius):
        dist_mtx, idx_mtx = self.tree.query(points.cpu().numpy(), max_k,
                                            distance_upper_bound=radius)

        return (torch.from_numpy(dist_mtx).float().to(self.device),
                torch.from_numpy(idx_mtx).long().to(self.device))


class Render:
    def __init__(self, mpoints):
        pass

    def query(self, qpoints):
        with context.current():
            draw = renderviz.Draw(renderviz.DrawMode.POINTbS, "vert.vert")

            draw['verts'] = mmodel.points
            draw['frags'] = qmodel.points

            draw()


class ProjectionNN:
    def __init__(self, kcam, rt_cam, points, device, image_size):

        # camera_points = Homogeneous(
        #     torch.from_numpy(rt_cam.world_to_cam)) @ points
        # image_points = kcam.project(camera_points)

        # width, height = image_size

        # image_points = image_points[:, :2].floor()
        # mask = ((image_points >= 0).all(1)
        #         & (image_points[:, 0] < width)
        #         & (image_points[:, 1] < height))

        # image_points = image_points[mask, :]

        # self.indexmap = IndexMap(width, height, image_points)
        # points = points.to(device)
        self.tree = Octree(points, 512)
        # import ipdb; ipdb.set_trace()

        self.kcam = kcam

    def query(self, points, k):
        #points = self.kcam.project(points)
        # return self.indexmap.query(points, k, 0.1)
        return self.tree.query2(points, k, 0.01)


def _main():
    import json
    import tenviz.io

    from fiontb.camera import RTCamera, KCamera

    model = tenviz.io.read_3dobject("_test/nn_model.ply").torch()
    live = tenviz.io.read_3dobject("_test/nn_live.ply").torch()

    with open('_test/frame-info.json', 'r') as file:
        frame_info = json.load(file)
        rt_cam = RTCamera.from_json(frame_info['rt_cam'])
        kcam = KCamera.from_json(frame_info['kcam']).torch()

    #import torch_geometric.nn as tgnn
    # N = tgnn.nearest(model.verts.to("cuda:0"), live.verts.to("cuda:0"))

    prof = Profile()
    prof.enable()

    tree = cKDTree(model.verts.numpy())
    tree.query(live.verts.numpy()[:2048,], 4, distance_upper_bound=0.01)

    model.verts = model.verts.to("cuda:0")
    live.verts = live.verts.to("cuda:0")[:2048, ]

    proj_nn = ProjectionNN(kcam, rt_cam, model.verts, "cuda:0", (640, 480))
    dist, idx = proj_nn.query(live.verts, 4)
    prof.disable()
    prof.dump_stats("profile.prof")

    # proj_nn = ProjectionNN(kcam, rt_cam, )


if __name__ == '__main__':
    _main()
