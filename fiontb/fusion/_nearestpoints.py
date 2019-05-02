import torch
from scipy.spatial import cKDTree


class KDTree:
    def __init__(self, points, device):
        self.tree = cKDTree(points.cpu(), balanced_tree=False)
        self.device = device

    def query(self, points):
        dist_mtx, idx_mtx = self.tree.query(points.cpu().numpy(), 9)

        return (torch.from_numpy(dist_mtx).float().to(self.device),
                torch.from_numpy(idx_mtx).long().to(self.device))

class Render:
    def __init__(self, mpoints):
        pass

    def query(self, qpoints):
        with context.current():
            draw = renderviz.Draw(renderviz.DrawMode.POINTS, "vert.vert")
            
            draw['verts'] = mmodel.points
            draw['frags'] = qmodel.points

            draw()
