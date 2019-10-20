from pathlib import Path
import math

import tenviz

from fiontb.data.ftb import load_ftb
from fiontb.data import set_cameras_to_start_at_eye
from fiontb.frame import FramePointCloud
from fiontb.surfel import SurfelModel
from fiontb.pose import ICPVerifier
from fiontb.pose.icp import MultiscaleICPOdometry
from fiontb.testing import prepare_frame
from fiontb.viz.show import show_pcls

from ..fusion import SurfelFusion

_LAST_FRAME = 2


def _test():
    test_data = Path(__file__).parent / "../../../../test-data/rgbd"

    dataset = load_ftb(test_data / "sample1")  # 20 frames
    set_cameras_to_start_at_eye(dataset)

    gl_context = tenviz.Context()

    model = SurfelModel(gl_context, 1024*1024*10, feature_size=3)

    fusion = SurfelFusion(model, normal_max_angle=math.radians(
        80), max_merge_distance=0.5, stable_conf_thresh=0)

    device = "cuda:0"
    for i in range(0, _LAST_FRAME):
        frame = dataset[i]

        frame, features = prepare_frame(
            frame, compute_normals=True, filter_depth=False, to_hsv=True)
        fpcl = FramePointCloud.from_frame(frame).to(device)

        stats = fusion.fuse(fpcl, fpcl.rt_cam, features.to(device))

    icp = MultiscaleICPOdometry([
        (0.25, 20, True),
        (0.5, 20, True),
        (1.0, 20, True)
    ])

    next_frame, next_features = prepare_frame(
        dataset[_LAST_FRAME], compute_normals=True, filter_depth=True, to_hsv=True)
    next_fpcl = FramePointCloud.from_frame(next_frame).to(device)
    next_features = next_features.to(device)

    fpcl, features = fusion.get_model_frame_pcl(flip=True)
    if True:
        fpcl.plot_debug(show=False)
        next_fpcl.plot_debug(show=False)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(features.transpose(1, 0).transpose(1, 2).cpu())
        plt.figure()
        plt.imshow(next_features.transpose(1, 0).transpose(1, 2).cpu())
        plt.show()

    result = icp.estimate(
        next_fpcl.kcam, next_fpcl.points, next_fpcl.mask,
        target_points=fpcl.points,
        target_normals=fpcl.normals,
        target_mask=fpcl.mask,
        source_feats=next_features,
        target_feats=features, geom_weight=.5, feat_weight=.5)

    print(ICPVerifier()(result))

    pcl0 = fpcl.unordered_point_cloud(world_space=True)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    align = fpcl.rt_cam.integrate(result.transform.to(device))
    pcl2 = pcl1.transform(align.cam_to_world.to(device))

    show_pcls([pcl0, pcl1, pcl2])


if __name__ == '__main__':
    _test()
