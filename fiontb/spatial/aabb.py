import torch


class AABB:
    @classmethod
    def from_points(cls, points):
        minv = points.min(0)[0].cpu()
        maxv = points.max(0)[0].cpu()

        return AABB(minv, maxv)

    def __init__(self, minv, maxv):
        self.min = torch.Tensor(minv)
        self.max = torch.Tensor(maxv)

    @property
    def center(self):
        return (self.min + self.max)*.5

    def is_inside(self, points, radius=None):
        if radius is None:
            minv = self.min.to(points.device)
            maxv = self.max.to(points.device)
            return ((points[:, 0] >= minv[0]) & (points[:, 0] <= maxv[0])
                    & (points[:, 1] >= minv[1]) & (points[:, 1] <= maxv[1])
                    & (points[:, 2] >= minv[2]) & (points[:, 2] <= maxv[2]))

        closest = self.closest_point(points)
        dists = torch.norm(points - closest, 2, 1)
        radius = radius * radius

        return dists < radius

    def closest_point(self, points):
        return points.max(self.min).min(self.max)

    def __str__(self):
        return "min: {},\nmax: {}".format(self.min, self.max)

    def __repr__(self):
        return str(self)


def subdivide_octo(box):
    center = box.center
    boxes = [
        AABB.from_points(torch.Tensor(
            [[box.min[0], box.min[1], box.min[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.min[0], box.min[1], box.max[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.min[0], box.max[1], box.max[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.min[0], box.max[1], box.min[2]], center])),

        AABB.from_points(torch.Tensor(
            [[box.max[0], box.min[1], box.min[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.max[0], box.min[1], box.max[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.max[0], box.max[1], box.max[2]], center])),
        AABB.from_points(torch.Tensor(
            [[box.max[0], box.max[1], box.min[2]], center]))]

    return boxes
