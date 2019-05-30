"""Workflow for testing fusion using surfels.
"""

import numpy as np
import open3d

import rflow

from odometry import FrameToFrameOdometry, ViewOdometry
from ui import ReconstructionLoop, MainLoop, RunMode
from fiontb.camera import RTCamera
import fiontb.fusion
import fiontb.pose.icp


class SurfelReconstructionStep(ReconstructionLoop):
    def __init__(self, surfels, surfels_lock, surfel_update_queue, odometry):
        self.surfels = surfels
        self.surfels_lock = surfels_lock
        self.surfel_update_queue = surfel_update_queue
        self.odometry = odometry
        self.count = 0

        self.fusion_ctx = fiontb.fusion.SurfelFusion(surfels)
        self.curr_rt_cam = RTCamera(np.eye(4, dtype=np.float32))

    def step(self, kcam, frame_points, live_pcl):
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

        live_pcl.transform(self.curr_rt_cam.cam_to_world)

        self.surfels_lock.acquire()
        surfel_update, surfel_removal = self.fusion_ctx.fuse(
            frame_points, live_pcl, kcam)
        self.surfels_lock.release()
        self.surfel_update_queue.put(
            (surfel_update.cpu(), surfel_removal.cpu()))

        self.count += 1


class SurfelFusion(rflow.Interface):
    def evaluate(self, resource, dataset, odometry):
        from cProfile import Profile
        sensor = fiontb.sensor.DatasetSensor(dataset)
        
        loop = MainLoop(sensor, SurfelReconstructionStep, resource.filepath, odometry,
                        max_frames=None, single_process=True, run_mode=RunMode.STEP)
        prof = Profile()
        prof.enable()
        loop.run()
        prof.disable()
        prof.dump_stats("surfel_fusion.prof")


class SuperDenseFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        fusion = fiontb.fusion.surfel.DensePCLFusion(3, 0.2)

        _main_loop(sensor, fusion, resource.filepath, False)


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

    g.fusion = SurfelFusion(rflow.FSResource("scene3.ply"))
    with g.fusion as args:
        args.dataset = ds_g.to_ftb
        args.odometry = g.frame_to_frame

    g.dense_fusion = SuperDenseFusion(rflow.FSResource("scene3.ply"))
    with g.dense_fusion as args:
        args.dataset = ds_g.to_ftb


class LoadFTB(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb

        return load_ftb(resource.filepath)


@rflow.graph()
def chair1(g):
    g.dataset = LoadFTB(rflow.FSResource("chair1"))

    g.fusion = SurfelFusion(rflow.FSResource("chair1.ply"))
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

    g.fusion = SurfelFusion(rflow.FSResource("lr0.ply"))
    with g.fusion as args:
        args.dataset = ds_g.lr0_to_ftb
        args.odometry = g.frame_to_frame

    g.dense_fusion = SuperDenseFusion(rflow.FSResource("lr0.ply"))
    with g.dense_fusion as args:
        args.dataset = ds_g.lr0_to_ftb


if __name__ == '__main__':
    rflow.command.main()
