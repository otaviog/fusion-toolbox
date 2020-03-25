"""Testing procedures for registration methods.
"""
from pathlib import Path

from tqdm import tqdm

from slamtb.metrics import (relative_rotational_error,
                            relative_translational_error)
from slamtb.frame import FramePointCloud
from slamtb.surfel import SurfelCloud
from slamtb.camera import RTCamera
from slamtb.registration.icp import ICPVerifier
from slamtb.viz import geoshow
from slamtb.testing import preprocess_frame, ColorSpace
from slamtb._utils import profile as _profile


def run_pair_test(icp, dataset, profile_file=None, filter_depth=True, blur=True,
                  color_space=ColorSpace.LAB, frame0_idx=0, frame1_idx=8,
                  device="cuda:0"):
    """Test with two frames.
    """

    frame_args = {
        'filter_depth': filter_depth,
        'blur': blur,
        'color_space': color_space
    }

    target_frame, target_feats = preprocess_frame(
        dataset[frame0_idx], **frame_args)
    source_frame, source_feats = preprocess_frame(
        dataset[frame1_idx], **frame_args)

    target_fpcl = FramePointCloud.from_frame(target_frame).to(device)
    source_fpcl = FramePointCloud.from_frame(source_frame).to(device)

    verifier = ICPVerifier()
    with _profile(Path(__file__).parent / str(profile_file),
                  profile_file is not None):
        for _ in range(1 if profile_file is None else 5):
            result = icp.estimate_frame(source_frame, target_frame,
                                        source_feats=source_feats.to(
                                            device),
                                        target_feats=target_feats.to(
                                            device),
                                        device=device)
        relative_rt = result.transform.cpu().float()

    if not verifier(result):
        print("Tracking failed")

    gt_traj = {0: target_fpcl.rt_cam, 1: source_fpcl.rt_cam}
    pred_traj = {0: RTCamera(), 1: RTCamera(relative_rt)}

    print("Translational error: ", relative_translational_error(
        gt_traj, pred_traj).item())
    print("Rotational error: ", relative_rotational_error(
        gt_traj, pred_traj).item())

    target_pcl = target_fpcl.unordered_point_cloud(world_space=False)
    source_pcl = source_fpcl.unordered_point_cloud(world_space=False)
    aligned_pcl = source_pcl.transform(relative_rt.to(device))

    print("Key 1 - toggle target PCL")
    print("Key 2 - toggle source PCL")
    print("Key 3 - toggle aligned source PCL")

    geoshow([target_pcl, source_pcl, aligned_pcl],
            title=icp.__class__.__name__, invert_y=True)


def run_pcl_pair_test(registration, dataset, profile_file=None, filter_depth=True, blur=True,
                      color_space=ColorSpace.LAB, frame0_idx=0, frame1_idx=8,
                      device="cuda:0"):
    """Run testing alignment using the `estimate_pcl` method.
    """

    frame_args = {
        'filter_depth': filter_depth,
        'blur': blur,
        'color_space': color_space
    }

    target_frame, target_features = preprocess_frame(dataset[frame0_idx], **frame_args)
    source_frame, source_features = preprocess_frame(dataset[frame1_idx], **frame_args)

    device = "cuda:0"

    target_pcl = SurfelCloud.from_frame(target_frame, features=target_features).to(device)
    source_pcl = SurfelCloud.from_frame(source_frame, features=source_features).to(device)

    with _profile(profile_file):
        result = registration.estimate_pcl(source_pcl, target_pcl,
                                           source_feats=source_pcl.features,
                                           target_feats=target_pcl.features)

    gt_traj = {0: target_frame.info.rt_cam, 1: source_frame.info.rt_cam}
    pred_traj = {0: RTCamera(), 1: RTCamera(result.transform)}

    print("Translational error: ", relative_translational_error(
        gt_traj, pred_traj).item())
    print("Rotational error: ", relative_rotational_error(
        gt_traj, pred_traj).item())

    print("Key 1 - toggle target PCL")
    print("Key 2 - toggle source PCL")
    print("Key 3 - toggle aligned source PCL")

    aligned_pcl = source_pcl.transform(result.transform)
    geoshow([target_pcl.as_point_cloud(),
             source_pcl.as_point_cloud(),
             aligned_pcl.as_point_cloud()], invert_y=True,
            title=registration.__class__.__name__)


def run_trajectory_test(icp, dataset, filter_depth=True, blur=True,
                        color_space=ColorSpace.LAB):
    """Trajectory test."""

    device = "cuda:0"

    verifier = ICPVerifier()

    frame_args = {
        'filter_depth': filter_depth,
        'blur': blur,
        'color_space': color_space
    }
    prev_frame, prev_features = preprocess_frame(
        dataset[0], **frame_args)

    pcls = [FramePointCloud.from_frame(
        prev_frame).unordered_point_cloud(world_space=False)]

    pose_accum = RTCamera()
    pred_traj = {0: pose_accum}
    gt_traj = {0: prev_frame.info.rt_cam}

    for i in tqdm(range(1, len(dataset))):
        next_frame, next_features = preprocess_frame(
            dataset[i], **frame_args)

        result = icp.estimate_frame(next_frame, prev_frame,
                                    source_feats=next_features.to(
                                        device),
                                    target_feats=prev_features.to(
                                        device),
                                    device=device)
        if not verifier(result):
            print("{} tracking fail".format(i))

        pose_accum = pose_accum.transform(
            result.transform.cpu().double())
        pred_traj[i] = pose_accum
        gt_traj[i] = next_frame.info.rt_cam

        pcl = FramePointCloud.from_frame(
            next_frame).unordered_point_cloud(world_space=False)
        pcl = pcl.transform(pose_accum.matrix.float())
        pcls.append(pcl)

        prev_frame, prev_features = next_frame, next_features

    print("Translational error: ", relative_translational_error(
        gt_traj, pred_traj).mean().item())
    print("Rotational error: ", relative_rotational_error(
        gt_traj, pred_traj).mean().item())

    geoshow(pcls, title=icp.__class__.__name__, invert_y=True)
