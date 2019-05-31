"""Benchmark kNN algorithms.
"""

import argparse
import time
import json
from pathlib import Path
from cProfile import Profile

from tabulate import tabulate

from scipy.spatial import cKDTree

import torch
from fiontb.fiontblib import Octree

import tenviz.io

from fiontb.camera import RTCamera, KCamera, Homogeneous
from fiontb.spatial.cindexmap import cIndexMap


def _main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--cprofile')
    args = parser.parse_args()

    here = Path(__file__).parent
    model = tenviz.io.read_3dobject(here / "nn_model.ply").torch()
    live = tenviz.io.read_3dobject(here / "nn_live.ply").torch()
    with open(here / 'frame-info.json', 'r') as file:
        frame_info = json.load(file)
        rt_cam = RTCamera.from_json(frame_info['rt_cam'])
        kcam = KCamera.from_json(frame_info['kcam'])

    prof = None
    if args.cprofile:
        prof = Profile()
        prof.enable()

    model_points = Homogeneous(torch.from_numpy(
        rt_cam.world_to_cam)) @ model.verts

    timings = []

    begin = time.perf_counter()
    indexmap = cIndexMap(model_points, kcam, 640, 480)
    end = time.perf_counter()
    timings.append(['IndexMap.build', end - begin])

    begin = time.perf_counter()
    dist_mtx, idx_mtx = indexmap.query(live.verts, 8)
    end = time.perf_counter()
    timings.append(['IndexMap.query', end - begin])

    begin = time.perf_counter()
    tree = cKDTree(model_points.numpy())
    end = time.perf_counter()
    timings.append(['cKDTree.build', end - begin])

    query_points = live.verts.numpy()
    begin = time.perf_counter()
    kdist_mtx, kidx_mtx = tree.query(query_points, 8)
    end = time.perf_counter()
    timings.append(['cKDTree.query', end - begin])

    import ipdb
    ipdb.set_trace()
    if prof is not None:
        prof.disable()
        prof.dump_stats(args.cprofile)

    print(tabulate(timings))


if __name__ == '__main__':
    _main()
