"""Workflow for testing fusion using surfels.
"""

import numpy as np
import open3d

import rflow

from odometry import FrameToFrameOdometry, ViewOdometry
from ui import MainLoop, RunMode
from fiontb.camera import RTCamera
import fiontb.fusion
import fiontb.pose.icp


class LoadFTB(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb

        return load_ftb(resource.filepath)


class ReconstructionStep:
    def __init__(self, fusion_ctx, odometry):
        self.fusion_ctx = fusion_ctx
        self.odometry = odometry
        self.count = 0
        self.curr_rt_cam = RTCamera(np.eye(4, dtype=np.float32))

    def step(self, kcam, frame_points):
        if self.odometry is not None:
            self.curr_rt_cam = RTCamera(
                np.array(self.odometry[self.count]['rt_cam'], dtype=np.float32))
        else:
            if False:
                fusion_pcl = self.fusion_ctx.get_odometry_model()
                if not fusion_pcl.is_empty():
                    relative_rt_cam = fiontb.pose.icp.estimate_icp_geo(
                        live_pcl, fusion_pcl)
                    self.curr_rt_cam.integrate(relative_rt_cam)
            else:
                self.curr_rt_cam = frame_points.rt_cam

        # eye = RTCamera(np.eye(4, dtype=np.float32))
        self.fusion_ctx.fuse(
            frame_points, kcam, self.curr_rt_cam)

        self.count += 1


class FusionTask(rflow.Interface):
    def evaluate(self, resource, dataset, odometry):
        from cProfile import Profile
        import torch
        import tenviz

        from fiontb.fusion.surfel import SurfelModel, SurfelFusion

        device = "cuda:0"
        torch.rand(4, 4).to(device) # init torch cuda
        sensor = fiontb.sensor.DatasetSensor(dataset)

        context = tenviz.Context(dataset[0].depth_image.shape[1],
                                 dataset[0].depth_image.shape[0])

        surfel_model = SurfelModel(context, 1024*1024*10)
        fusion_ctx = SurfelFusion(surfel_model)
        step = ReconstructionStep(fusion_ctx, odometry)

        loop = MainLoop(sensor, surfel_model, step,
                        max_frames=None, run_mode=RunMode.STEP)
        prof = Profile()
        prof.enable()
        loop.run()
        prof.disable()
        prof.dump_stats("surfel_fusion.prof")


@rflow.graph()
def scene1(g):
    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))

    g.fusion = FusionTask(rflow.FSResource("scene1.ply"))
    with g.fusion as args:
        args.dataset = g.dataset
        args.odometry = None


@rflow.graph()
def scene3(g):
    ds_g = rflow.open_graph("../../test-data/rgbd/scene3", "sample")

    g.frame_to_frame = FrameToFrameOdometry(
        rflow.FSResource("frame2frame.json"))
    with g.frame_to_frame as args:
        args.dataset = ds_g.to_ftb

    g.view_frame_to_frame = ViewOdometry()
    with g.view_frame_to_frame as args:
        args.dataset = ds_g.to_ftb
        args.odometry = g.frame_to_frame

    g.fusion = FusionTask(rflow.FSResource("scene3.ply"))
    with g.fusion as args:
        args.dataset = ds_g.to_ftb
        args.odometry = g.frame_to_frame


@rflow.graph()
def chair1(g):
    g.dataset = LoadFTB(rflow.FSResource("chair1"))

    g.fusion = FusionTask(rflow.FSResource("chair1.ply"))
    with g.fusion as args:
        args.dataset = g.dataset
        args.odometry = None


@rflow.graph()
def iclnuim(g):
    ds_g = rflow.open_graph("../../test-data/rgbd/iclnuim", "iclnuim")

    g.frame_to_frame = FrameToFrameOdometry(
        rflow.FSResource("lr0-frame2frame.json"))
    with g.frame_to_frame as args:
        args.dataset = ds_g.lr0_to_ftb

    g.view_frame_to_frame = ViewOdometry()
    with g.view_frame_to_frame as args:
        args.dataset = ds_g.lr0_to_ftb
        args.odometry = g.frame_to_frame

    g.fusion = FusionTask(rflow.FSResource("lr0.ply"))
    with g.fusion as args:
        args.dataset = ds_g.lr0_to_ftb
        args.odometry = g.frame_to_frame


if __name__ == '__main__':
    rflow.command.main()
