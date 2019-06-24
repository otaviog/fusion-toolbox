import torch
import torch.nn.functional as F


def batch(tensor, batch_size):
    for iteration in range(0, tensor.size(0), batch_size):
        area = slice(iteration,
                     min(iteration + batch_size, tensor.size(0)-1))
        yield area, tensor[area, :]


class BruteNNSearch:
    def __init__(self, points):
        self.points = points

    def query(self, query, n):
        dists = torch.zeros((query.size(0),), dtype=query.dtype)
        idxs = torch.zeros((query.size(0), ), dtype=torch.int64)

        for i, (area, q) in enumerate(batch(query, 256)):
            mul = self.points @ q.transpose(1, 0)
            ldist, lidx = torch.min(mul, 0)
            dists[area] = ldist
            idxs[area] = lidx

        return dists, idxs
