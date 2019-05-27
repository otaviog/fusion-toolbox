from collections import defaultdict

import torch

from .aabb import AABB, subdivide_octo


class OctNode:
    @classmethod
    def create_non_leaf(cls, points, indices, box, leaf_num_points):
        box = AABB.from_points(points[indices])
        boxes = subdivide_octo(box)
        children = [None]*8
        for i, curr_box in enumerate(boxes):
            mask = curr_box.is_inside(points[indices])
            inside_count = mask.sum()
            if inside_count == 0:
                continue

            inside_indices = indices[mask.nonzero().squeeze().view(-1)]

            if inside_count <= leaf_num_points:
                children[i] = OctNode(points, curr_box, inside_indices)
            else:
                children[i] = OctNode.create_non_leaf(points, inside_indices,
                                                      curr_box, leaf_num_points)

        return OctNode(points, box, None, children)

    def __init__(self, model_points, box, inside_indices, children=None):
        self.model_points = model_points
        self.box = box
        self.indices = inside_indices
        self.children = children

    def query(self, qpoints, radius, which_qpoints,
              dist_result, idx_result):
        if self.children is None:
            dists = self.model_points[self.indices] @ qpoints[which_qpoints].transpose(
                1, 0)

            dists, indices = dists.sort(0)
            indices = self.indices[indices]

            for idx in which_qpoints:
                idx = int(idx.item())
                dist_result[idx].append(dists)
                idx_result[idx].append(indices)

            return

        for child in self.children:
            if child is None:
                continue

            mask = child.box.is_inside(qpoints[which_qpoints], radius)
            if mask.sum() == 0:
                continue

            child.query(qpoints, radius, which_qpoints[mask].view(-1),
                        dist_result, idx_result)


class OctTree:
    def __init__(self, points, leaf_num_points=16):
        self.points = points
        self.box = AABB.from_points(points)

        self.root = OctNode.create_non_leaf(
            points, torch.arange(0, points.size(0), dtype=torch.int64,
                                 device=points.device),
            self.box, leaf_num_points)

    def query(self, points, radius, k):
        dist_result = defaultdict(list)
        idx_result = defaultdict(list)

        self.root.query(points, radius, torch.arange(0, points.size(0), dtype=torch.int64),
                        dist_result, idx_result)
        
        return dist_result
