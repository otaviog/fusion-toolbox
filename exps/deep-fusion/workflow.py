import matplotlib.pyplot as plt
import torch

import rflow


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


class RTResidual(rflow.Interface):
    @staticmethod
    def _show_pcl_render(pcl, kcam, rt_cam):
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

    @staticmethod
    def _get_geom_residual(p0, n0, idxmap0,
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

    @staticmethod
    def _get_residual(feats0, idxmap0, feats1, idxmap1):

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

    def evaluate(self, dataset, frame_num0, frame_num1):
        from torch.nn.functional import upsample
        import torchvision
        import torchvision.transforms as transforms

        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        # model = torchvision.models.vgg16(pretrained=True)
        # model = torchvision.models.resnet152(pretrained=True)
        model = torchvision.models.densenet201(pretrained=True)
        # A = torch.rand(1, 3, 640, 480)
        # import ipdb; ipdb.set_trace()

        model = model.eval()
        model = model.features[:3]
        vgg_norm = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        with torch.no_grad():
            feat_img0 = model(
                vgg_norm(frame0.rgb_image).unsqueeze(0))
            feat_img0 = upsample(feat_img0, scale_factor=2)
            feat_img0 = feat_img0.squeeze().transpose(0, 2).transpose(0, 1)
            feats0 = feat_img0.squeeze().reshape(-1, 64)

            feat_img1 = model(
                vgg_norm(frame1.rgb_image).unsqueeze(0))
            feat_img1 = upsample(feat_img1, scale_factor=2)
            feat_img1 = feat_img1.squeeze().transpose(0, 2).transpose(0, 1)
            feats1 = feat_img1.squeeze().reshape(-1, 64)

        import ipdb
        ipdb.set_trace()

        pcl0 = FramePointCloud(frame0).unordered_point_cloud()
        pcl1 = FramePointCloud(frame1).unordered_point_cloud()

        indexmap = IndexMap(640, 480)

        idxmap0 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl0.points, frame0.info.kcam).clone().cpu()
        idxmap1 = indexmap.raster(Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ pcl1.points, frame1.info.kcam).clone().cpu()

        geom_residual = RTResidual._get_geom_residual(pcl0.points, pcl0.normals, idxmap0,
                                                      pcl1.points, pcl1.normals, idxmap1)
        residual = RTResidual._get_residual(feats0, idxmap0, feats1, idxmap1)

        plt.figure()
        render0 = RTResidual._show_pcl_render(
            pcl0, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame0 render")
        plt.imshow(render0)

        plt.figure()
        render1 = RTResidual._show_pcl_render(
            pcl1, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame1 render")
        plt.imshow(render1)

        plt.figure()
        render10 = RTResidual._show_pcl_render(
            pcl1, frame0.info.kcam, frame0.info.rt_cam)
        plt.title("Frame1 render from camera0")
        plt.imshow(render10)

        plt.figure()
        plt.title("Deep Residual")
        plt.imshow(residual)

        plt.figure()
        plt.title("Geom Residual")
        plt.imshow(geom_residual)

        plt.show()


class TFResidual(rflow.Interface):
    @staticmethod
    def _get_residual(feats0, idxmap0, feats1, idxmap1):

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

        import ipdb; ipdb.set_trace()

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
        import ipdb; ipdb.set_trace()
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

        residual = RTResidual._get_residual(feats0, idxmap0, feats1, idxmap1)

        plt.figure()
        render0 = RTResidual._show_pcl_render(
            pcl0, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame0 render")
        plt.imshow(render0)

        plt.figure()
        render1 = RTResidual._show_pcl_render(
            pcl1, frame1.info.kcam, frame1.info.rt_cam)
        plt.title("Frame1 render")
        plt.imshow(render1)

        plt.figure()
        render10 = RTResidual._show_pcl_render(
            pcl1, frame0.info.kcam, frame0.info.rt_cam)
        plt.title("Frame1 render from camera0")
        plt.imshow(render10)

        plt.figure()
        plt.title("Deep Residual")
        plt.imshow(residual)

        plt.figure()
        plt.title("Geom Residual")
        plt.imshow(geom_residual)

        plt.show()


@rflow.graph()
def residual_map(g):
    g.dataset = LoadFTB(rflow.FSResource("scenenn-objs/scene1"))
    g.gt_mesh = LoadMesh(rflow.FSResource("scenenn-objs/scene1.ply"))

    g.intermediate = GenerateIntermediateFusion(
        rflow.FSResource("interm-fusion.torch"))
    with g.intermediate as args:
        args.dataset = g.dataset
        args.max_frame_num = 80

    g.vgg_residual = RTResidual()
    with g.vgg_residual as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 40

    g.tfeat_residual = TFResidual()
    with g.tfeat_residual as args:
        args.dataset = g.dataset
        args.frame_num0 = 0
        args.frame_num1 = 40
