"""Workflow for testing fusion using surfels.
"""

# pylint: disable=missing-docstring

import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms

import rflow

from fiontb.camera import RTCamera
import fiontb.fusion
import fiontb.pose.open3d

from odometry import FrameToFrameOdometry, ViewOdometry
from ui import MainLoop, RunMode


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
        self.curr_rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
        self.use_gt = True
        self.prev_frame = None

        model = torchvision.models.vgg16(pretrained=True)
        model = model.eval()
        self.model = model.features[:4]
        self.vgg_norm = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])


    def step(self, frame, frame_points):
        if self.odometry is not None:
            self.curr_rt_cam = RTCamera(
                torch.tensor(self.odometry[self.count]['rt_cam'], dtype=np.float32))
        elif self.use_gt:
            self.curr_rt_cam = frame_points.rt_cam
        elif False:
            if self.fusion_ctx.pose_indexmap.is_rasterized:
                source_frame = self.fusion_ctx.pose_indexmap.to_frame(
                    frame.info)

                if False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.imshow(source_frame.depth_image)

                    plt.figure()
                    plt.imshow(source_frame.rgb_image)

                    plt.show()

                relative_cam = fiontb.pose.open3d.estimate_odometry(
                    source_frame, frame)
                self.curr_rt_cam = self.curr_rt_cam.integrate(relative_cam)
        elif True:
            if self.prev_frame is not None:
                relative_cam = fiontb.pose.open3d.estimate_odometry(
                    self.prev_frame, frame)
                self.curr_rt_cam = self.curr_rt_cam.integrate(relative_cam)

        self.prev_frame = frame

        # with torch.no_grad():
        if False:
            feat = self.model(self.vgg_norm(frame.rgb_image).unsqueeze(0))
            feat = feat.squeeze().transpose(0, 2).transpose(0, 1)
            feat = feat.squeeze().reshape(-1, 64)

        self.fusion_ctx.fuse(
            frame_points, self.curr_rt_cam,
            # features=feat.to("cuda:0")
        )

        self.count += 1


class FusionTask(rflow.Interface):
    def evaluate(self, resource, dataset, odometry, gt_mesh):
        from cProfile import Profile

        import tenviz
        from tenviz.io import write_3dobject

        from fiontb.fusion.surfel import SurfelModel, SurfelFusion

        device = "cuda:0"
        torch.rand(4, 4).to(device)  # init torch cuda
        sensor = fiontb.sensor.DatasetSensor(dataset)

        context = tenviz.Context(dataset[0].depth_image.shape[1],
                                 dataset[0].depth_image.shape[0])

        surfel_model = SurfelModel(context, 1024*1024*5, "cuda:0", 64)
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
                torch.from_numpy(frame.depth_image),
                frame_pcl.depth_image, depth_scale=frame.info.depth_scale)

            frame_pcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                                 frame_pcl.depth_image).cpu()
            with torch.no_grad():
                feat = model(vgg_norm(frame.rgb_image).unsqueeze(0))
                feat = feat.squeeze().transpose(0, 2).transpose(0, 1)
                feat = feat.squeeze().reshape(-1, 64)
                feats.append(feat)

            frames.append(frame_pcl)

        context = tenviz.Context(640, 480)
        surfel_model = SurfelModel(context, 1024*1024*4, "cuda:0")
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
