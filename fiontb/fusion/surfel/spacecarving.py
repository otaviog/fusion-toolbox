from pathlib import Path

import torch
import matplotlib.pyplot as plt

import tenviz

from ._ckernels import surfel_cave_free_space

_SHADER_DIR = Path(__file__).parent / "shaders"


class SpaceCarvingContext:
    def __init__(self, surfel_model):
        with surfel_model.context.current():
            self.render_surfels_prg = tenviz.DrawProgram(
                tenviz.DrawMode.Points,
                vert_shader=_SHADER_DIR / "render_model.vert",
                frag_shader=_SHADER_DIR / "render_model.frag")

            self.render_surfels_prg['in_point'] = surfel_model.points
            self.render_surfels_prg['in_normal'] = surfel_model.normals
            self.render_surfels_prg['in_conf'] = surfel_model.confs
            self.render_surfels_prg['in_time'] = surfel_model.times
            self.render_surfels_prg['in_mask'] = surfel_model.active_mask_gl

            self.render_surfels_prg['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.render_surfels_prg['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.render_surfels_prg['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

            self.stable_fb = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBFloat,  # pos
                1: tenviz.FramebufferTarget.RGBFloat,  # normal
                2: tenviz.FramebufferTarget.RGBInt32  # indexes
            })

        self.surfel_model = surfel_model

    def carve(self, rt_cam, width, height, kcam, min_depth, max_depth,
              stable_conf_thresh, time, debug=False):
        context = self.surfel_model.context

        proj = tenviz.projection_from_kcam(
            kcam.matrix, min_depth, max_depth)
        proj_matrix = torch.from_numpy(proj.to_matrix()).float()

        context.set_clear_color(0, 0, 0, 0)

        with context.current():
            self.render_surfels_prg['StableThresh'] = torch.tensor(
                [stable_conf_thresh], dtype=torch.float32)
            self.render_surfels_prg['Time'] = torch.tensor(
                [time], dtype=torch.int32)

        context.render(proj_matrix,
                       torch.from_numpy(rt_cam.opengl_view_cam).float(),
                       self.stable_fb,
                       [self.render_surfels_prg], width, height)

        fbs = self.stable_fb.get_attachs()
        with context.current():
            stable_positions = fbs[0].to_tensor()
            stable_normals = fbs[1].to_tensor()
            stable_idxs = fbs[2].to_tensor()[:, :, 0].contiguous()

        if debug:
            plt.figure()
            plt.title("pos")
            plt.imshow(stable_positions[:, :, 2])
            plt.figure()
            plt.title("normals")
            plt.imshow(stable_normals)
            plt.figure()
            plt.title("stable_idxs")
            plt.imshow(stable_idxs)
            plt.show()

        with context.current():
            self.render_surfels_prg['StableThresh'] = torch.tensor(
                [-1.0], dtype=torch.float32)
            self.render_surfels_prg['Time'] = torch.tensor(
                [-1], dtype=torch.int32)

        context.render(proj_matrix,
                       torch.from_numpy(rt_cam.opengl_view_cam).float(),
                       self.stable_fb,
                       [self.render_surfels_prg], width, height)

        fbs = self.stable_fb.get_attachs()
        with context.current():
            view_positions = fbs[0].to_tensor()
            view_normals = fbs[1].to_tensor()
            view_idxs = fbs[2].to_tensor()[:, :, 0].contiguous()

        if debug:
            plt.figure()
            plt.title("pos")
            plt.imshow(view_positions[:, :, 2])
            plt.figure()
            plt.title("normals")
            plt.imshow(view_normals)
            plt.figure()
            plt.title("stable_idxs")
            plt.imshow(view_idxs)
            plt.show()

        import ipdb
        ipdb.set_trace()

        surfel_cave_free_space(stable_positions.to("cuda:0"),
                               stable_idxs.to("cuda:0"),
                               view_positions.to("cuda:0"),
                               view_idxs.to("cuda:0"),
                               self.surfel_model.active_mask, 16)


def _test():
    import numpy as np
    import tenviz.io

    from fiontb.viz.surfelrender import SurfelRender
    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud

    test_data = Path(__file__).parent / "_test"

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))

    rt_cam = RTCamera.create_from_pos_quat(
        # 72
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681
    )

    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    radii = torch.full((model_size, ), 0.05, dtype=torch.float)
    confs = torch.full((model_size, ), 15, dtype=torch.float)
    times = torch.full((model_size, ), 5, dtype=torch.int32)

    model_cloud = SurfelCloud(model.verts, model.colors, model.normals,
                              radii, confs, times, "cpu")
    model_cloud.to("cuda:0")

    surfel_model.add_surfels(model_cloud)

    np.random.seed(10)
    space_vl_sample = np.random.choice(model.verts.size(0), 100)
    verts = (model.verts[space_vl_sample]
             + (torch.from_numpy(rt_cam.center()).float() -
                model.verts[space_vl_sample])
             * torch.rand(100, 1))
    space_vl_cloud = SurfelCloud(verts,
                                 model_cloud.colors[space_vl_sample],
                                 model_cloud.normals[space_vl_sample],
                                 radii[space_vl_sample],
                                 confs[space_vl_sample],
                                 torch.full((100, ), 3, dtype=torch.int32),
                                 "cpu")
    space_vl_cloud.to("cuda:0")
    surfel_model.add_surfels(space_vl_cloud)
    surfel_model.update_active_mask_gl()

    surfel_render = SurfelRender(surfel_model)
    viewer = ctx.viewer([surfel_render], tenviz.CameraManipulator.WASD)
    # viewer.set_camera_matrix(rt_cam.opengl_view_cam)
    while True:
        q = chr(viewer.wait_key(1) & 0xff).lower()
        if q == 'q':
            break

    carving_ctx = SpaceCarvingContext(surfel_model)
    carving_ctx.carve(rt_cam, 640, 480, kcam, 0.01, 10.0,
                      10, 5, debug=False)

    surfel_model.update_active_mask_gl()

    # viewer.set_camera_matrix(rt_cam.opengl_view_cam)
    viewer.show(1)


if __name__ == "__main__":
    _test()
