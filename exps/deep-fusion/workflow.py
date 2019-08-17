import matplotlib.pyplot as plt
import torch

import rflow

from fiontb.pose.open3d import estimate_odometry as estimate_odometry_open3d


class LoadFTB(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_ftb

        return load_ftb(resource.filepath)


class LoadMesh(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from tenviz.io import read_3dobject

        return read_3dobject(resource.filepath).torch()


class LoadOdometry(rflow.Interface):
    def evaluate(self, resource):
        return self.load(resource)

    def load(self, resource):
        from fiontb.data.ftb import load_trajectory

        return load_trajectory(resource.filepath)


class GenerateIntermediateFusion(rflow.Interface):
    def evaluate(self, resource, dataset, max_frame_num):
        import torch

        import tenviz

        from fiontb.fusion.surfel import SurfelModel, SurfelFusion
        from fiontb.frame import FramePointCloud, estimate_normals
        from fiontb.filtering import bilateral_filter_depth_image

        context = tenviz.Context()
        model = SurfelModel(context, 1024*1024*5)
        fusion = SurfelFusion(model)

        for frame_num in range(max_frame_num):
            frame = dataset[frame_num]
            frame_pcl = FramePointCloud(frame)

            mask = frame_pcl.fg_mask.to("cuda:0")
            filtered_depth_image = bilateral_filter_depth_image(
                torch.from_numpy(frame.depth_image).to("cuda:0"),
                mask, depth_scale=frame.info.depth_scale)

            frame_pcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                                 mask).cpu()

            fusion.fuse(frame_pcl, frame.info.rt_cam)

        surfel_cloud = model.to_surfel_cloud()
        torch.save(surfel_cloud, resource.filepath)

        return surfel_cloud

    def load(self, resource):
        import torch
        return torch.load(resource.filepath)


class ViewAlign(rflow.Interface):
    @staticmethod
    def _render_pcl(pcl, kcam, rt_cam):
        import tenviz

        ctx = tenviz.Context(640, 480)
        with ctx.current():
            scene = [tenviz.create_point_cloud(pcl.points, pcl.colors)]
            framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBUint8})

        proj_matrix = torch.from_numpy(tenviz.projection_from_kcam(
            kcam.matrix, 0.01, 100.0).to_matrix()).float()

        ctx.render(proj_matrix, rt_cam.opengl_view_cam, framebuffer,
                   scene)

        with ctx.current():
            render = framebuffer[0].to_tensor()

        return render.cpu()

    def evaluate(self, dataset, frame_num0, frame_num1):
        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        pcl0 = FramePointCloud(frame0).unordered_point_cloud()
        pcl1 = FramePointCloud(frame1).unordered_point_cloud()

        plt.figure()
        render0 = ViewAlign._render_pcl(
            pcl0, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame0 render")
        plt.imshow(render0)

        plt.figure()
        render1 = ViewAlign._render_pcl(
            pcl1, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame1 render")
        plt.imshow(render1)

        plt.figure()
        render10 = ViewAlign._render_pcl(
            pcl1, frame0.info.kcam, frame0.info.rt_cam)
        plt.title("Frame1 render from camera0")
        plt.imshow(render10)

        plt.show()


def _get_odometry(frame0, frame1, odo):
    if odo == 'gt':
        return frame1.info.rt_cam

    if odo == 'open3d':
        transform = estimate_odometry_open3d(frame0, frame1)
        rt_cam = frame0.info.rt_cam.integrate(transform)

        return rt_cam
    return None


class GeometricResidual(rflow.Interface):
    @staticmethod
    def _get_residual(p0, n0, idxmap0,
                      p1, n1, idxmap1):
        residual = torch.full(idxmap0.size(), -0.01)
        for row in range(idxmap0.size(0)):
            for col in range(idxmap0.size(1)):
                idx0 = idxmap0[row, col]
                if idx0 < 0:
                    continue

                idx1 = idxmap1[row, col]
                if idx1 < 0:
                    continue

                # residual[row, col] = (p1[idx1] - p0[idx0]).dot(n0[idx0])
                residual[row, col] = (p1[idx1] - p0[idx0]).norm()

        return residual

    def evaluate(self, dataset, frame_num0, frame_num1, odo='gt'):
        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        from fiontb.data.ftb import write_ftb

        write_ftb("sample1", [dataset[i] for i in range(0, 100, 5)])
        pcl0 = FramePointCloud(frame0).unordered_point_cloud()
        pcl1 = FramePointCloud(frame1).unordered_point_cloud()

        rt_cam = _get_odometry(frame0, frame1, odo)

        indexmap = IndexMap(640, 480)

        idxmap0 = indexmap.raster(Homogeneous(
            rt_cam.world_to_cam) @ pcl0.points, frame0.info.kcam).clone().cpu()
        idxmap1 = indexmap.raster(Homogeneous(
            rt_cam.world_to_cam) @ pcl1.points, frame1.info.kcam).clone().cpu()

        geom_residual = GeometricResidual._get_residual(pcl0.points, pcl0.normals, idxmap0,
                                                        pcl1.points, pcl1.normals, idxmap1)
        plt.figure()
        plt.title("Geometric residual {} {} - {}".format(odo,
                                                         frame_num0, frame_num1))
        plt.imshow(geom_residual)
        plt.show()


def _get_feat_residual(feats0, idxmap0, feats1, idxmap1):
    residual = torch.full(idxmap0.size(), -0.01)
    for row in range(idxmap0.size(0)):
        for col in range(idxmap0.size(1)):
            idx0 = idxmap0[row, col]
            if idx0 < 0:
                continue

            idx1 = idxmap1[row, col]
            if idx1 < 0:
                continue

            feat0 = feats0[idx0, :]
            feat1 = feats1[idx1, :]

            residual[row, col] = (feat1 - feat0).norm()

    return residual


class ConvResidual(rflow.Interface):

    def evaluate(self, dataset, frame_num0, frame_num1, net, odo='gt'):
        from torch.nn.functional import upsample
        import torchvision as tv
        
        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        net_map = {
            'vgg16': (tv.models.vgg16, 4, 1.0),
            'resnet152': (tv.models.resnet152, 1, 1.0),
            'densenet201': (tv.models.densenet201, 3, 2.0)
        }
        model_factory, layer, scale = net_map[net]

        model = model_factory(pretrained=True)

        model = model.eval()
        model = model.features[:layer]
        vgg_norm = tv.transforms.Compose([tv.ToTensor(),
                                          tv.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])

        with torch.no_grad():
            feat_img0 = model(
                vgg_norm(frame0.rgb_image).unsqueeze(0))
            feat_img0 = upsample(feat_img0, scale_factor=scale)

            feat_img0 = feat_img0.squeeze().transpose(0, 2).transpose(0, 1)
            feats0 = feat_img0.squeeze().reshape(-1, 64)

            feat_img1 = model(
                vgg_norm(frame1.rgb_image).unsqueeze(0))

            feat_img1 = upsample(feat_img1, scale_factor=scale)
            feat_img1 = feat_img1.squeeze().transpose(0, 2).transpose(0, 1)
            feats1 = feat_img1.squeeze().reshape(-1, 64)

        fpcl0 = FramePointCloud(frame0)
        feats0 = feats0[fpcl0.fg_mask.flatten()]
        pcl0 = fpcl0.unordered_point_cloud()

        fpcl1 = FramePointCloud(frame1)
        feats1 = feats1[fpcl1.fg_mask.flatten()]
        pcl1 = fpcl1.unordered_point_cloud()

        rt_cam = _get_odometry(frame0, frame1, odo)

        indexmap = IndexMap(640, 480)
        idxmap0 = indexmap.raster(Homogeneous(
            rt_cam.world_to_cam) @ pcl0.points, frame0.info.kcam).clone().cpu()
        idxmap1 = indexmap.raster(Homogeneous(
            rt_cam.world_to_cam) @ pcl1.points, frame1.info.kcam).clone().cpu()

        residual = _get_feat_residual(feats0, idxmap0, feats1, idxmap1)

        plt.figure()
        plt.title("{} Residual {} {} - {}".format(net,
                                                  odo, frame_num0, frame_num1))
        plt.imshow(residual)
        plt.show()


class TFResidual(rflow.Interface):
    @staticmethod
    def _extract_feats(frame, tfeat):
        import cv2
        from torch.utils.data import DataLoader
        gray = cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2GRAY)
        gray = torch.from_numpy(gray).float()

        gray_pad = torch.zeros(480+31, 640+31)
        gray_pad[:480, :640] = gray
        gray = gray_pad
        patches = gray.unfold(0, 32, 1).unfold(1, 32, 1)

        patches = patches.reshape(-1, 1, 32, 32)

        patch_loader = DataLoader(patches, batch_size=32, drop_last=False)
        feats = []

        for patch in patch_loader:
            feat = tfeat(patch.to("cuda:0"))
            feats.append(feat)

        feats = torch.cat(feats).cpu()
        return feats

    def evaluate(self, dataset, frame_num0, frame_num1):
        from pathlib import Path
        import tfeat_model

        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        tfeat = tfeat_model.TNet()
        tfeat.load_state_dict(torch.load(
            str(Path(tfeat_model.__file__).parent / "pretrained-models/tfeat-liberty.params")))

        tfeat = tfeat.eval()
        tfeat = tfeat.to("cuda:0")

        with torch.no_grad():
            feats0 = TFResidual._extract_feats(frame0, tfeat)
            feats1 = TFResidual._extract_feats(frame1, tfeat)

        fpcl0 = FramePointCloud(frame0)
        feats0 = feats0[fpcl0.fg_mask.flatten()]
        pcl0 = fpcl0.unordered_point_cloud()

        fpcl1 = FramePointCloud(frame1)
        feats1 = feats1[fpcl1.fg_mask.flatten()]
        pcl1 = fpcl1.unordered_point_cloud()

        indexmap = IndexMap(640, 480)
        idxmap0 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl0.points, frame0.info.kcam).clone().cpu()
        idxmap1 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl1.points, frame1.info.kcam).clone().cpu()

        residual = _get_feat_residual(feats0, idxmap0, feats1, idxmap1)

        plt.figure()
        plt.title("TF Residual frames {} - {}".format(frame_num0, frame_num1))
        plt.imshow(residual)
        plt.show()


class PointNetResidual(rflow.Interface):
    @staticmethod
    def _preproc(points):
        import numpy as np
        points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
        points = points / dist  # scale

        T = torch.from_numpy(points).unsqueeze(0).transpose(2, 1)
        return T

    @staticmethod
    def _extract_feats(frame, tfeat):
        import cv2
        from torch.utils.data import DataLoader
        gray = cv2.cvtColor(frame.rgb_image, cv2.COLOR_RGB2GRAY)
        gray = torch.from_numpy(gray).float()

        gray_pad = torch.zeros(480+31, 640+31)
        gray_pad[:480, :640] = gray
        gray = gray_pad
        patches = gray.unfold(0, 32, 1).unfold(1, 32, 1)

        patches = patches.reshape(-1, 1, 32, 32)

        patch_loader = DataLoader(patches, batch_size=32, drop_last=False)
        feats = []

        for patch in patch_loader:
            feat = tfeat(patch.to("cuda:0"))
            feats.append(feat)

        feats = torch.cat(feats).cpu()
        return feats

    def evaluate(self, dataset, frame_num0, frame_num1):
        from pathlib import Path
        from pointnet.model import PointNetfeat, PointNetCls, PointNetDenseCls

        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        model = PointNetDenseCls(k=4)
        model.load_state_dict(torch.load('seg_model_Chair_24.pth'))
        model = model.feat
        model.eval()

        fpcl0 = FramePointCloud.from_frame(frame0)
        pcl0 = fpcl0.unordered_point_cloud()

        fpcl1 = FramePointCloud.from_frame(frame1)
        pcl1 = fpcl1.unordered_point_cloud()

        with torch.no_grad():
            feats0 = model(PointNetResidual._preproc(pcl0.points.numpy()))[
                0].squeeze().transpose(1, 0)
            feats1 = model(PointNetResidual._preproc(pcl1.points.numpy()))[
                0].squeeze().transpose(1, 0)

        indexmap = IndexMap(640, 480)
        idxmap0 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl0.points, frame0.info.kcam).clone().cpu()
        idxmap1 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl1.points, frame1.info.kcam).clone().cpu()

        residual = _get_feat_residual(
            feats0, idxmap0, feats1, idxmap1)

        plt.figure()
        plt.title("PointNet Residual frames {} - {}".format(frame_num0, frame_num1))
        plt.imshow(residual)

        plt.figure()
        plt.title("Geom Residual")
        plt.imshow(geom_residual)

        plt.show()


@rflow.graph()
def residuals(g):
    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))
    g.gt_mesh = LoadMesh(rflow.FSResource("scenenn-objs/scene1.ply"))
    g.frame2frame_odo = LoadOdometry(
        rflow.FSResource("scenenn-objs/scene1/odometry.json"))

    g.intermediate = GenerateIntermediateFusion(
        rflow.FSResource("interm-fusion.torch"))
    with g.intermediate as args:
        args.dataset = g.dataset
        args.max_frame_num = 80

    g.geom = GeometricResidual()
    with g.geom as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25

    g.geom_open3d = GeometricResidual()
    with g.geom_open3d as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25
        args.odo = 'open3d'

    g.conv_densenet = ConvResidual()
    with g.conv_densenet as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25
        args.net = "densenet201"

    g.conv_vgg16 = ConvResidual()
    with g.conv_vgg16 as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25
        args.net = "vgg16"

    g.conv_vgg16_open3d = ConvResidual()
    with g.conv_vgg16_open3d as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25
        args.net = "vgg16"
        args.odo = 'open3d'

    g.tfeat = TFResidual()
    with g.tfeat as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25

    g.pointnet = PointNetResidual()
    with g.pointnet as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 25
