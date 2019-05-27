"""Benchmark kNN algorithms.
"""

import argparse
import time
from cProfile import Profile

from tabulate import tabulate

from scipy.spatial import cKDTree

import torch
from fiontb.fiontblib import Octree


def _main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cprofile')
    args = parser.parse_args()

    torch.manual_seed(10)

    prof = None
    if args.cprofile:
        prof = Profile()
        prof.enable()

    model_points = torch.rand(205000, 3)
    query_points = torch.rand(2048, 3)
    timings = []

    model_points_np = model_points.numpy()
    
    tree = cKDTree(model_points_np)
    begin = time.perf_counter()
    tree.query(query_points, 10, distance_upper_bound=0.01)
    end = time.perf_counter()

    timings.append(['cKDTree', end - begin])

    model_points_gpu = model_points.to("cuda:0")


    octree = Octree(model_points_gpu, 1024)
    begin = time.perf_counter()
    octree.query(query_points, 10, 0.01)
    end = time.perf_counter()

    timings.append(['Octree', end - begin])

    if prof is not None:
        prof.disable()
        prof.dump_stats(args.cprofile)

    print(tabulate(timings))


if __name__ == '__main__':
    _main()
