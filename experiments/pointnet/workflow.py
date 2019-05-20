import rflow

import numpy as np
import torch

import matplotlib.pyplot as plt

from fiontb.camera import KCamera


def preproc(points):
    points = points - np.expand_dims(np.mean(points, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)), 0)
    points = points / dist  # scale

    T = torch.from_numpy(points).unsqueeze(0).transpose(2, 1)
    return T


class GeneratePointClouds(rflow.Interface):

    def evaluate(self, model, kcam):
        import torch

        import tenviz
        import tenviz.io
        import tenviz.camera

        from fiontb.camera import Homogeneous

        geo = tenviz.io.read_3dobject(model).torch()

        bmin = geo.verts.min(0)[0]
        bmax = geo.verts.max(0)[0]

        center = (bmax + bmin)*.5
        radius = torch.norm(bmax - center, 2).item()

        proj = tenviz.projection_from_kcam(kcam.matrix, 0.01, 5)
        proj_mtx = proj.to_matrix()

        torch.manual_seed(10)
        context = tenviz.Context(640, 480)
        with context.current():
            draw = tenviz.DrawProgram(tenviz.DrawMode.Triangles,
                                      "point-sampling.vert",
                                      "point-sampling.frag")
            draw['in_position'] = geo.verts
            draw['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            draw['Modelview'] = tenviz.MatPlaceholder.Modelview

            draw.indices.from_tensor(geo.faces)

            framebuffer = tenviz.create_framebuffer({0: tenviz.FramebufferTarget.RGBFloat,
                                                     1: tenviz.FramebufferTarget.RUint8,
                                                     2: tenviz.FramebufferTarget.RGBAUint8})

        pcls = []
        i = True
        for elevation, azimuth in [(90, 0.1), (90, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)]:
            view = tenviz.camera.get_bounded_sphere_view(
                elevation, azimuth, center.numpy(), radius, proj)

            #proj_mtx = tenviz.perspective_proj(45, 0.01, 5, 1.0).to_matrix()
            #view = np.eye(4)
            #view[2, 3] = -2
            #view[2, 2] = -1
            context.render(torch.from_numpy(proj_mtx).float(),
                           torch.from_numpy(view).float(), framebuffer, [draw])

            with context.current():
                renders = framebuffer.get_attachs()
                positions = renders[0].to_tensor()
                selected_pos = renders[1].to_tensor()
                # rgb = renders[2].to_tensor()

            sampling = positions[selected_pos == 1, :]
            idxs = np.random.choice(
                sampling.size(0), 2500)
            sampling = sampling[idxs]
            if i:
                sampling += torch.rand(sampling.shape)*0.1
                i = False

            pcls.append((view, sampling))
        return pcls


class ViewPCLs(rflow.Interface):
    def evaluate(self, pcl_list):
        import tenviz
        import torch

        context = tenviz.Context(640, 480)
        with context.current():
            scene = []
            for view, pcl in pcl_list[:1]:
                tv_pcl = tenviz.create_point_cloud(
                    pcl, torch.tensor([1.0, 1.0, 1.0]))
                if view is not None:
                    tv_pcl.transform = np.linalg.inv(view)
                scene.append(tv_pcl)

        viewer = context.viewer(scene)
        viewer.reset_view()
        while True:
            key = viewer.wait_key(1)
            if key < 0:
                break


class ExtractFeatures(rflow.Interface):
    def evaluate(self, pcl_list, ref_model_filepath):
        import tenviz.io
        from pointnet.model import PointNetfeat, PointNetCls

        ref_geo = tenviz.io.read_3dobject(ref_model_filepath)

        #model = PointNetfeat(global_feat=True, feature_transform=False)
        model = PointNetCls(k=16)
        model.load_state_dict(torch.load('cls_model_149.pth'))
        model = model.feat
        model.eval()
        view, pcl = pcl_list[0]

        feats = []

        torch.manual_seed(10)
        with torch.no_grad():
            for noise_level in [0.0, 0.01, 0.1, 1.0]:
                feats.append([])
                for view, pcl in pcl_list:
                    pcl = pcl + torch.rand(pcl.shape)*noise_level
                    pred, _, _ = model(preproc(pcl.numpy()))
                    feats[-1].append(pred)

        with torch.no_grad():
            idxs = np.random.choice(
                ref_geo.verts.shape[0], 2500)

            ref_feat, _, _ = model(preproc(ref_geo.verts[idxs]))

        for nfeat in feats:
            pcl_feats = torch.stack(nfeat, 0).squeeze()
            dists = torch.cdist(ref_feat, pcl_feats)
            print(dists)

        return None


class Opt1(rflow.Interface):
    def evaluate(self, pcl_list, ref_model_filepath):
        import tenviz.io
        from pointnet.model import PointNetfeat, PointNetCls

        ref_geo = tenviz.io.read_3dobject(ref_model_filepath)

        model = PointNetCls(k=16)
        model.load_state_dict(torch.load('cls_model_149.pth'))
        model = model.feat
        model.eval()
        view, pcl = pcl_list[0]

        with torch.no_grad():
            idxs = np.random.choice(
                ref_geo.verts.shape[0], 2500)

            ref_feat, _, _ = model(preproc(ref_geo.verts[idxs]))
            #ref_feat, _, _ = model(preproc(pcl_list[1][1].numpy()))

        I = preproc(pcl.numpy())
        for i in range(300):
            I = I.requires_grad_()
            pred, _, _ = model(I)
            residual = (ref_feat - pred).pow(2).sum().sqrt() 
            residual.backward()
            print(i, residual.item())
            R = I.grad.clone()
            I = I - 0.005*R.float()
            I = I.detach()
        
        I = I.transpose(1, 2)
        I = I.squeeze()
        I = I.clone()
        # import ipdb; ipdb.set_trace()
        return [(None, I)]


@rflow.graph()
def exp1(g):
    g.pcls = GeneratePointClouds()
    with g.pcls as args:
        args.model = '../../test-data/bunny/bun_zipper_res4.ply'
        args.kcam = KCamera(np.array([[356.769928, 0.0, 251.563446],
                                      [0.0, 430.816498, 237.563446],
                                      [0.0, 0.0, 1.0]]))

    g.view_pcls = ViewPCLs()
    with g.view_pcls as args:
        args.pcl_list = g.pcls

    g.extract_features = ExtractFeatures()
    with g.extract_features as args:
        args.ref_model_filepath = "../../test-data/bunny/bun_zipper.ply"
        args.pcl_list = g.pcls


@rflow.graph()
def exp2(g):
    g.pcls = GeneratePointClouds()
    with g.pcls as args:
        args.model = 'model1.obj'
        args.kcam = KCamera(np.array([[356.769928, 0.0, 251.563446],
                                      [0.0, 430.816498, 237.563446],
                                      [0.0, 0.0, 1.0]]))

    g.view_pcls = ViewPCLs()
    with g.view_pcls as args:
        args.pcl_list = g.pcls

    g.extract_features = ExtractFeatures()
    with g.extract_features as args:
        args.ref_model_filepath = 'model2.obj'
        args.pcl_list = g.pcls

    g.extract_feature_e = ExtractFeatures()
    with g.extract_feature_e as args:
        args.ref_model_filepath = 'desk.obj'
        args.pcl_list = g.pcls


    g.opt1 = Opt1()
    with g.opt1 as args:
        args.ref_model_filepath = "model2.obj"
        args.pcl_list = g.pcls

    g.view = ViewPCLs()
    with g.view as args:
        args.pcl_list = g.opt1


@rflow.graph()
def exp3(g):
    g.pcls = GeneratePointClouds()
    with g.pcls as args:
        args.model = '../../test-data/bunny/bun_zipper_res4.ply'
        args.kcam = KCamera(np.array([[356.769928, 0.0, 251.563446],
                                      [0.0, 430.816498, 237.563446],
                                      [0.0, 0.0, 1.0]]))

    g.view_pcls = ViewPCLs()
    with g.view_pcls as args:
        args.pcl_list = g.pcls

    g.opt1 = Opt1()
    with g.opt1 as args:
        args.ref_model_filepath = "../../test-data/bunny/bun_zipper.ply"
        args.pcl_list = g.pcls

    g.view = ViewPCLs()
    with g.view as args:
        args.pcl_list = g.opt1









