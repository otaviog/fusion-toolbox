from pathlib import Path

import torch
from tqdm import tqdm
import quaternion
from tenviz.pose import Pose

from fiontb.metrics import (relative_rotational_error,
                            relative_translational_error)
from fiontb.frame import FramePointCloud
from fiontb.camera import RTCamera
from fiontb.registration.icp import ICPVerifier
from fiontb.viz import geoshow
from fiontb.testing import preprocess_frame, ColorSpace
from fiontb._utils import profile as _profile
from fiontb.data.ftb import load_ftb


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
    prev_frame, target_feats = preprocess_frame(
        dataset[frame0_idx], **frame_args)
    next_frame, source_feats = preprocess_frame(
        dataset[frame1_idx], **frame_args)

    prev_fpcl = FramePointCloud.from_frame(prev_frame).downsample(.5).to(device)
    next_fpcl = FramePointCloud.from_frame(next_frame).downsample(.5).to(device)

    verifier = ICPVerifier()

    with _profile(Path(__file__).parent / str(profile_file),
                  profile_file is not None):
        for _ in range(1 if profile_file is None else 5):
            result = icp.estimate_frame(next_frame, prev_frame,
                                        source_feats=source_feats.to(
                                            device),
                                        target_feats=target_feats.to(
                                            device),
                                        device=device)
        relative_rt = result.transform.cpu().float()

    if not verifier(result):
        print("Tracking failed")

    gt_traj = {0: prev_fpcl.rt_cam, 1: next_fpcl.rt_cam}
    pred_traj = {0: RTCamera(), 1: RTCamera(relative_rt)}

    print("Translational error: ", relative_translational_error(
        gt_traj, pred_traj).item())
    print("Rotational error: ", relative_rotational_error(
        gt_traj, pred_traj).item())

    pcl0 = prev_fpcl.unordered_point_cloud(world_space=False)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl2 = pcl1.transform(relative_rt.to(device))

    print("Key 1 - toggle target PCL")
    print("Key 2 - toggle source PCL")
    print("Key 3 - toggle aligned source PCL")

    geoshow([pcl0, pcl1, pcl2], title=icp.__class__.__name__, invert_y=True)


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
