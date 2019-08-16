"""Workflow for testing fusion using surfels.
"""

# pylint: disable=missing-docstring
import open3d
import torch

import rflow

from fiontb.camera import RTCamera
import fiontb.fusion
import fiontb.pose.open3d


class LoadFTB(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb

        return load_ftb(resource.filepath)

import matplotlib.pyplot as plt

class _Odometry:
    def __init__(self, mode, fusion_ctx):
        self.use_gt = False
        self.prev_frame = None
        self.mode = mode
        self.fusion_ctx = fusion_ctx

    def estimate(self, frame):
        if self.mode == "frame-to-frame":
            if self.prev_frame is not None:
                relative_cam = fiontb.pose.open3d.estimate_odometry(
                    self.prev_frame, frame)
            else:
                relative_cam = torch.eye(4, dtype=torch.float32)

            self.prev_frame = frame
            return relative_cam
        elif self.mode == "frame-to-model":
            if not self.fusion_ctx.pose_indexmap.is_rasterized:
                return torch.eye(4, dtype=torch.float32)
            
            source_frame = self.fusion_ctx.pose_indexmap.to_frame(
                frame.info)

            if False:
                plt.figure()
                plt.title("Model Depth")
                plt.imshow(source_frame.depth_image)
                
                plt.figure()
                plt.title("Model RGB")
                plt.imshow(source_frame.rgb_image)

                plt.figure()
                plt.title("Frame RGB")
                plt.imshow(frame.rgb_image)

                plt.figure()
                plt.title("Frame Depth")
                plt.imshow(frame.depth_image)
                
                plt.show()

            return fiontb.pose.open3d.estimate_odometry(
                source_frame, frame)
        elif self.mode == "ground-truth":
            if self.prev_frame is not None:
                relative_cam = (frame.info.rt_cam.cam_to_world
                                @ self.prev_frame.info.rt_cam.world_to_cam)

            else:
                relative_cam = frame.info.rt_cam.cam_to_world

            self.prev_frame = frame
            return relative_cam


class FusionTask(rflow.Interface):
    def evaluate(self, resource, dataset, gt_mesh):
        from cProfile import Profile

        import tenviz
        from tenviz.io import write_3dobject

        from fiontb.frame import FramePointCloud, estimate_normals
        from fiontb.filtering import bilateral_filter_depth_image
        from fiontb.fusion.surfel import SurfelModel, SurfelFusion
        from fiontb.sensor import DatasetSensor
        from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode

        device = "cuda:0"
        torch.rand(4, 4).to(device)  # init torch cuda

        sensor = DatasetSensor(dataset)

        context = tenviz.Context(dataset[0].depth_image.shape[1],
                                 dataset[0].depth_image.shape[0])

        surfel_model = SurfelModel(context, 1024*1024*5, "cuda:0")
        fusion_ctx = SurfelFusion(surfel_model)

        sensor_ui = FrameUI("Frame Control")
        rec_ui = SurfelReconstructionUI(surfel_model, RunMode.STEP, inverse=True)

        device = "cuda:0"
        rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
        odometry = _Odometry("frame-to-model", fusion_ctx)

        prof = Profile()
        prof.enable()
        for _ in rec_ui:
            frame = sensor.next_frame()
            if frame is None:
                continue

            live_fpcl = FramePointCloud.from_frame(frame)
            live_fpcl = live_fpcl.to(device)

            filtered_depth_image = bilateral_filter_depth_image(
                torch.from_numpy(frame.depth_image).to(device),
                live_fpcl.mask, depth_scale=frame.info.depth_scale)

            live_fpcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                                 live_fpcl.mask)

            frame.normal_image = live_fpcl.normals.cpu()

            sensor_ui.update(frame)

            relative_cam = odometry.estimate(frame)

            rt_cam = rt_cam.integrate(relative_cam)

            stats = fusion_ctx.fuse(live_fpcl, rt_cam)
            print(stats)

        prof.disable()
        prof.dump_stats("surfel_fusion.prof")

        # final_model = surfel_model.to_surfel_cloud()
        # write_3dobject(resource.filepath, final_model.points,
        # normals=final_model.normals,
        # colors=final_model.colors)

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
        vgg_norm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
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


class DeepFusionTask(rflow.Interface):
    def evaluate(self, resource, dataset, gt_mesh):
        from cProfile import Profile

        import torchvision

        import tenviz
        from tenviz.io import write_3dobject

        from fiontb.frame import FramePointCloud, estimate_normals
        from fiontb.filtering import bilateral_filter_depth_image
        from fiontb.fusion.surfel import SurfelModel, SurfelFusion
        from fiontb.sensor import DatasetSensor
        from fiontb.ui import FrameUI, SurfelReconstructionUI, RunMode

        device = "cuda:0"
        torch.rand(4, 4).to(device)  # init torch cuda

        sensor = DatasetSensor(dataset)

        context = tenviz.Context(dataset[0].depth_image.shape[1],
                                 dataset[0].depth_image.shape[0])

        surfel_model = SurfelModel(context, 1024*1024*2, "cuda:0", 64)
        fusion_ctx = SurfelFusion(surfel_model)

        sensor_ui = FrameUI("Frame Control")
        rec_ui = SurfelReconstructionUI(surfel_model, RunMode.STEP)

        device = "cuda:0"
        rt_cam = RTCamera(torch.eye(4, dtype=torch.float32))
        odometry = _Odometry("ground-truth", fusion_ctx)

        model = torchvision.models.vgg16(pretrained=True)
        model = model.eval().features[:4]
        vgg_norm = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

        prof = Profile()
        prof.enable()
        for _ in rec_ui:
            frame = sensor.next_frame()
            if frame is None:
                continue

            live_fpcl = FramePointCloud(frame)
            mask = live_fpcl.depth_mask.to(device)

            filtered_depth_image = bilateral_filter_depth_image(
                torch.from_numpy(frame.depth_image).to(device),
                mask, depth_scale=frame.info.depth_scale)

            live_fpcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                                 mask).cpu()

            frame.normal_image = live_fpcl.normals

            sensor_ui.update(frame)

            relative_cam = odometry.estimate(frame)
            rt_cam = rt_cam.integrate(relative_cam)

            with torch.no_grad():
                feats = model(
                    vgg_norm(frame.rgb_image).unsqueeze(0)).squeeze()
                feats = feats.transpose(0, 2).transpose(0, 1).reshape(-1, 64)

            fusion_ctx.fuse(live_fpcl, rt_cam, feats)

        prof.disable()
        prof.dump_stats("surfel_fusion.prof")

        final_model = surfel_model.to_surfel_cloud()
        write_3dobject(resource.filepath, final_model.points,
                       normals=final_model.normals,
                       colors=final_model.colors)

    def load(self, resource):
        pass


@rflow.graph()
def scene1(g):
    from fiontb.nodes.processing import (LoadMesh, MeshToPCL)
    from fiontb.nodes.evaluation import evaluation_graph

    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))
    g.dataset.show = False

    g.gt_mesh = LoadMesh(rflow.FSResource("scenenn-objs/scene1/scene1.ply"))
    g.gt_mesh.show = False

    g.gt_pcl = MeshToPCL()
    g.gt_pcl.args.mesh_geo = g.gt_mesh
    g.gt_pcl.show = False

    g.debug = FusionDebug()
    with g.debug as args:
        args.dataset = g.dataset
        args.gt_mesh = g.gt_mesh

    g.fusion = FusionTask(rflow.FSResource("scene1.ply"))
    with g.fusion as args:
        args.dataset = g.dataset
        args.gt_mesh = g.gt_mesh

    evaluation_graph(g, g.fusion, g.gt_mesh, g.gt_pcl, init_mtx=torch.eye(4))


@rflow.graph()
def scene1_deep(g):
    from fiontb.nodes.processing import (LoadMesh, MeshToPCL)
    from fiontb.nodes.evaluation import evaluation_graph

    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))
    g.dataset.show = False

    g.gt_mesh = LoadMesh(rflow.FSResource("scenenn-objs/scene1/scene1.ply"))
    g.gt_mesh.show = False

    g.gt_pcl = MeshToPCL()
    g.gt_pcl.args.mesh_geo = g.gt_mesh
    g.gt_pcl.show = False

    g.fusion = DeepFusionTask(rflow.FSResource("scene1-deep.ply"))
    with g.fusion as args:
        args.dataset = g.dataset
        args.gt_mesh = g.gt_mesh

    evaluation_graph(g, g.fusion, g.gt_mesh,
                     g.gt_pcl, init_mtx=torch.eye(4))


if __name__ == '__main__':
    rflow.command.main()
