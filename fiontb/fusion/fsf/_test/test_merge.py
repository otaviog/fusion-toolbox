from pathlib import Path
from cProfile import Profile

from scipy.spatial.ckdtree import cKDTree
import tenviz
import torch

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.viz.surfelrender import show_surfels
from fiontb.surfel import SurfelCloud, SurfelModel
from fiontb.testing import prepare_frame

from ..merge import Merge


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 20 frames
    set_cameras_to_start_at_eye(dataset)

    device = "cpu:0"

    scale = .5
    frame0, features0 = prepare_frame(
        dataset[0].clone(), scale=scale, filter_depth=False, compute_normals=True)
    frame1, features1 = prepare_frame(dataset[19].clone(), scale=scale, filter_depth=False,
                                      compute_normals=True)

    gl_context = tenviz.Context()
    model = SurfelModel(gl_context, 1024*1024*2)
    model.add_surfels(SurfelCloud.from_frame(frame0, time=0, world_space=True,
                                             features=features0).to(device), update_gl=True)
    prev_model = model.clone()

    target_surfels, global_indices = model.to_surfel_cloud()
    tree = cKDTree(target_surfels.positions.squeeze().cpu().numpy())

    source_surfels = SurfelCloud.from_frame(frame1, features=features1,
                                            world_space=True)
    _, knn_index = tree.query(
        source_surfels.positions.cpu().numpy(), k=5, distance_upper_bound=5e-3)
    knn_index = torch.from_numpy(knn_index).to(device)
    merge = Merge()

    merge(knn_index, source_surfels, global_indices, model, update_gl=True)

    show_surfels(gl_context, [prev_model, model])


if __name__ == '__main__':
    _test()
