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
    def _get_residual(feats0, feat_img1, idxmap):
        import torch

        residual = torch.zeros(idxmap.size())
        search_pos = [(0, 0), (-1, 0), (-1, -1), (0, -1),
                      (1, -1), (1, 0), (1, 1), (0, 1)]
        for row in range(idxmap.size(0)):
            for col in range(idxmap.size(1)):
                found = -1
                for i, j in ((row + k, col + w) for k, w in search_pos):
                    if (i < 0 or i >= idxmap.size(0)
                            or j < 0 or j >= idxmap.size(1)):
                        continue

                    idx = idxmap[i, j]
                    if idx >= 0:
                        found = idx

                if found < 0:
                    break

                feat1 = feat_img1[:, row, col]
                feat0 = feats0[found, ]

                residual[row, col] = (feat1 - feat0).norm()

        return residual

    def evaluate(self, dataset, frame_num0, frame_num1):
        import torch
        import torchvision
        import torchvision.transforms as transforms

        import matplotlib.pyplot as plt
        from fiontb.frame import FramePointCloud
        from fiontb.spatial.indexmap import IndexMap
        from fiontb.camera import Homogeneous

        frame0 = dataset[frame_num0]
        frame1 = dataset[frame_num1]

        model = torchvision.models.vgg16(pretrained=True)
        model = model.eval()
        model = model.features[:4]
        vgg_norm = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

        with torch.no_grad():
            feat_img0 = model(vgg_norm(frame0.rgb_image).unsqueeze(0))
            feats0 = feat_img0.squeeze().reshape(-1, 64)

            feat_img1 = model(
                vgg_norm(frame1.rgb_image).unsqueeze(0)).squeeze()

        local_pcl0 = FramePointCloud(frame0).unordered_point_cloud()
        points0 = Homogeneous(
            frame1.info.rt_cam.world_to_cam) @ local_pcl0.points

        indexmap = IndexMap(640, 480)
        indexmap.raster(points0, frame1.info.kcam)
        idxmap = indexmap.indexmap
        plt.imshow(idxmap.cpu())
        plt.show()
        residual = RTResidual._get_residual(feats0, feat_img1, idxmap)
        plt.imshow(residual)
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
        args.frame_num1 = 0
