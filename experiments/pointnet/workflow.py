import rflow

import numpy as np

import matplotlib.pyplot as plt


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
        for elevation, azimuth in [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)]:
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
                rgb = renders[2].to_tensor()

            sampling = positions[selected_pos == 1, :]
            idxs = np.random.choice(
                sampling.size(0), 2500) # int(sampling.size(0)*0.05))
            sampling = sampling[idxs]
            pcls.append((view, sampling))

        return pcls


class ViewPCLs(rflow.Interface):
    def evaluate(self, pcl_list):
        import tenviz
        import torch

        context = tenviz.Context(640, 480)
        with context.current():
            scene = []
            for view, pcl in pcl_list:
                tv_pcl = tenviz.create_point_cloud(
                    pcl, torch.tensor([1.0, 1.0, 1.0]))
                tv_pcl.transform = np.linalg.inv(view)
                scene.append(tv_pcl)

        viewer = context.viewer(scene)
        viewer.reset_view()
        while True:
            key = viewer.wait_key(1)
            if key < 0:
                break


class ExtractFeatures(rflow.Interface):
    def evaluate(self, pcl_list):
        import torch
        from pointnet.model import PointNetCls

        model = PointNetCls(k=16)
        model.load_state_dict(torch.load('cls_model_149.pth'))
        model.eval()
        view, pcl = pcl_list[0]

        feats = []
        with torch.no_grad():
            for view, pcl in pcl_list:
                pred = model(pcl.view(1, 3, -1))            
                feats.append(pred)

        return None


@rflow.graph()
def exp1(g):
    from fiontb.camera import KCamera

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
        args.pcl_list = g.pcls
