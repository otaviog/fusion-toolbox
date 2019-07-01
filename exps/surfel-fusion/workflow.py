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

        self.fusion_ctx.fuse(
            frame_points, self.curr_rt_cam)

        self.count += 1


class FusionTask(rflow.Interface):
    def evaluate(self, resource, dataset, odometry, gt_mesh):
        from cProfile import Profile

        import torch

        import tenviz
        from tenviz.io import write_3dobject

        from fiontb.fusion.surfel import SurfelModel, SurfelFusion

        device = "cuda:0"
        torch.rand(4, 4).to(device)  # init torch cuda
        sensor = fiontb.sensor.DatasetSensor(dataset)

        context = tenviz.Context(dataset[0].depth_image.shape[1],
                                 dataset[0].depth_image.shape[0])

        surfel_model = SurfelModel(context, 1024*1024*50)
        fusion_ctx = SurfelFusion(surfel_model)
        step = ReconstructionStep(fusion_ctx, odometry)

        loop = MainLoop(sensor, surfel_model, step, fusion_ctx,
                        max_frames=None, run_mode=RunMode.STEP,
                        # gt_mesh=gt_mesh.torch()
                        )
        prof = Profile()
        prof.enable()
        loop.run()
        prof.disable()
        prof.dump_stats("surfel_fusion.prof")
        final_model = surfel_model.to_surfel_cloud()
        final_model.to("cpu")
        write_3dobject(resource.filepath, final_model.points,
                       normals=final_model.normals,
                       colors=final_model.colors)

    def load(self, resource):
        pass


class FusionDebug(rflow.Interface):
    def evaluate(self, dataset, odometry, gt_mesh):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        import tenviz

        from fiontb.frame import FramePointCloud, estimate_normals
        from fiontb.filtering import bilateral_filter_depth_image
        from fiontb.fusion.surfel import SurfelModel, SurfelFusion
        from fiontb.viz.surfelrender import show_surfels

        model = torchvision.models.vgg16(pretrained=True)
        model = model.eval()
        model = model.features[:4]
        vgg_norm = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        device = "cuda:0"
        torch.rand(4, 4).to(device)  # init torch cuda

        start_frame = 0

        frames = []
        feats = []
        for frame_num in range(4):
            frame = dataset[start_frame + frame_num]
            frame_pcl = FramePointCloud(frame)
            filtered_depth_image = bilateral_filter_depth_image(
                frame.depth_image.astype(np.float32)*0.001, frame_pcl.depth_mask)
            filtered_depth_image = (
                filtered_depth_image*1000.0).numpy().astype(np.int32)
            frame_pcl.normals = compute_normals(filtered_depth_image, frame.info,
                                                frame_pcl.depth_mask)
            with torch.no_grad():
                feat = model(vgg_norm(frame.rgb_image).unsqueeze(0))
                feats.append(feat.squeeze().reshape(-1, 64))

            frames.append(frame_pcl)

        context = tenviz.Context(640, 480)
        surfel_model = SurfelModel(context, 1024*1024*4, "cuda:0", 64)
        fusion_ctx = SurfelFusion(surfel_model)

        stats = fusion_ctx.fuse(
            frames[0], frames[0].rt_cam, feats[0])
        f0_res = surfel_model.compact()
        print(stats)

        stats = fusion_ctx.fuse(
            frames[1], frames[1].rt_cam, feats[1])
        f1_res = surfel_model.compact()
        print(stats)

        stats = fusion_ctx.fuse(
            frames[2], frames[2].rt_cam, feats[2])
        f2_res = surfel_model.compact()
        print(stats)

        stats = fusion_ctx.fuse(
            frames[3], frames[3].rt_cam, feats[3])
        f3_res = surfel_model.compact()
        print(stats)

        with context.current():
            max_conf = f3_res.confs.to_tensor().max()

        show_surfels(context, [f0_res, f1_res, f2_res,
                               f3_res], max_conf=max_conf.item(), max_time=4)


@rflow.graph()
def scene1(g):
    import torch

    from fiontb.nodes.processing import (LoadMesh, MeshToPCL)
    from fiontb.nodes.evaluation import evaluation_graph

    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))
    g.dataset.show = False
    g.gt_mesh = LoadMesh(rflow.FSResource("scenenn-objs/scene1.ply"))
    g.gt_mesh.show = False

    g.fusion = FusionTask(rflow.FSResource("scene1.ply"))
    with g.fusion as args:
        args.dataset = g.dataset
        args.odometry = None
        args.gt_mesh = g.gt_mesh

    g.debug = FusionDebug()
    with g.debug as args:
        args.dataset = g.dataset
        args.odometry = None
        args.gt_mesh = g.gt_mesh

    g.gt_pcl = MeshToPCL()
    g.gt_pcl.args.mesh_geo = g.gt_mesh
    g.gt_pcl.show = False

    evaluation_graph(g, g.fusion, g.gt_mesh, g.gt_pcl, init_mtx=torch.eye(4))


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
