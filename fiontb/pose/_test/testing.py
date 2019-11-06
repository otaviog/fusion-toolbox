from pathlib import Path

import torch
from tqdm import tqdm
import quaternion

from fiontb.metrics import (absolute_translational_error,
                            rotational_error)
from fiontb.frame import FramePointCloud
from fiontb.camera import RTCamera
from fiontb.pose.icp import ICPVerifier
from fiontb.viz.show import show_pcls
from fiontb.testing import prepare_frame, ColorMode
from fiontb._utils import profile as _profile
from fiontb.data.ftb import load_ftb

from tenviz.pose import Pose


def evaluate(gt_cam0, gt_cam1, relative_rt):
    gt_pose = Pose.from_matrix(gt_cam0.matrix.inverse() @ gt_cam1.matrix)
    pred_pose = Pose.from_matrix(relative_rt)

    gt_trans = torch.tensor(gt_pose.get_translation())
    pred_trans = torch.tensor(pred_pose.get_translation())

    gt_rot = quaternion.from_float_array(gt_pose.get_quaternion())
    pred_rot = quaternion.from_float_array(pred_pose.get_quaternion())

    print("Translation error: ", (gt_trans - pred_trans).abs().mean().item())
    print("Rotation error: ", (gt_rot - pred_rot).norm())


def run_pair_test(icp, dataset, profile_file=None, filter_depth=True, blur=True,
                  color_mode=ColorMode.HSV, frame0_idx=0, frame1_idx=8):
    device = "cuda:0"
    frame_args = {
        'filter_depth': filter_depth,
        'blur': blur,
        'color_mode': color_mode
    }
    prev_frame, target_feats = prepare_frame(dataset[frame0_idx], **frame_args)
    next_frame, source_feats = prepare_frame(dataset[frame1_idx], **frame_args)

    prev_fpcl = FramePointCloud.from_frame(prev_frame).to(device)
    next_fpcl = FramePointCloud.from_frame(next_frame).to(device)

    verifier = ICPVerifier()

    with _profile(Path(__file__).parent / str(profile_file), profile_file is not None):
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

    evaluate(prev_fpcl.rt_cam, next_fpcl.rt_cam,
             relative_rt)
    pcl0 = prev_fpcl.unordered_point_cloud(world_space=False)
    pcl1 = next_fpcl.unordered_point_cloud(world_space=False)
    pcl2 = pcl1.transform(relative_rt.to(device))

    # pcl0.colors[:] = torch.tensor([0, 0, 255])
    # pcl2.colors[:] = torch.tensor([255, 0, 0])
    show_pcls([pcl0, pcl1, pcl2])


def evaluate_trajectory(dataset, traj):
    base = dataset.get_info(0).rt_cam.matrix.inverse()

    gt_traj = {}
    for i in range(len(dataset)):
        info = dataset.get_info(i)
        gt_traj[i] = info.rt_cam.integrate(base)

    print("ATE-RMSE: ", absolute_translational_error(gt_traj, traj).mean().item())
    print("Rotation error: ", rotational_error(gt_traj, traj).mean().item())


def run_trajectory_test(icp, dataset, filter_depth=True, blur=True,
                        to_hsv=True, to_gray=False):
    device = "cuda:0"

    verifier = ICPVerifier()

    frame_args = {
        'filter_depth': filter_depth,
        'blur': blur,
        'to_hsv': to_hsv,
        'to_gray': to_gray
    }
    prev_frame, prev_features = prepare_frame(
        dataset[0], **frame_args)

    pcls = [FramePointCloud.from_frame(
        prev_frame).unordered_point_cloud(world_space=False)]

    accum_pose = RTCamera(dtype=torch.double)
    traj = {0: accum_pose}

    for i in tqdm(range(1, len(dataset))):
        # for i in range(1, 75):
        next_frame, next_features = prepare_frame(
            dataset[i], **frame_args)

        result = icp.estimate_frame(next_frame, prev_frame,
                                    source_feats=next_features.to(
                                        device),
                                    target_feats=prev_features.to(
                                        device),
                                    device=device)
        if not verifier(result):
            print("{} tracking fail".format(i))
        accum_pose = accum_pose.integrate(result.transform.cpu().double())
        traj[i] = accum_pose.clone()

        pcl = FramePointCloud.from_frame(
            next_frame).unordered_point_cloud(world_space=False)
        pcl = pcl.transform(accum_pose.matrix.float())
        pcls.append(pcl)

        prev_frame, prev_features = next_frame, next_features
    evaluate_trajectory(dataset, traj)
    show_pcls(pcls)
