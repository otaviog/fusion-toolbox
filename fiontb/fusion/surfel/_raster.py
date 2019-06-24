from pathlib import Path

import torch
import matplotlib.pyplot as plt

import tenviz

_SHADER_DIR = Path(__file__).parent / "shaders"


class LiveSurfelRaster:
    def __init__(self, context):

        with context.current():
            self.program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "render_live.vert",
                _SHADER_DIR / "render_live.frag")

            self.points = tenviz.buffer_create()
            self.normals = tenviz.buffer_create()

            self.program['in_point'] = self.points
            self.program['in_normal'] = self.normals
            
            self.program['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.program['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview
            self.program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            self.framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBFloat,
                1: tenviz.FramebufferTarget.RGBAFloat,
                2: tenviz.FramebufferTarget.RGBInt32
            })

        self.context = context

    def raster(self, surfel_cloud, proj_matrix, width, height):
        view_mtx = torch.eye(4, dtype=torch.float32)
        view_mtx[2, 2] = -1

        with self.context.current():
            self.points.from_tensor(surfel_cloud.points)
            self.normals.from_tensor(surfel_cloud.normals)

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix, view_mtx, self.framebuffer,
                            [self.program], width, height)

    @property
    def pos_tex(self):
        return self.framebuffer[0]

    @property
    def normal_rad_tex(self):
        return self.framebuffer[1]

    @property
    def idx_tex(self):
        return self.framebuffer[2]

    def show_debug(self, title, debug=True):
        if not debug:
            return

        with self.context.current():
            pos = self.pos_tex.to_tensor()
            normals = self.normal_rad_tex.to_tensor()
            idxs = self.idx_tex.to_tensor()

        plt.figure()
        plt.title("{} - Positions".format(title))
        plt.imshow(pos[:, :, 2].cpu())

        plt.figure()
        plt.title("{} - Normals".format(title))
        plt.imshow(normals.cpu())

        plt.figure()
        plt.title("{} - Indices".format(title))
        plt.imshow(idxs[:, :, 0].cpu())

        plt.show()

class GlobalSurfelRaster:
    def __init__(self, surfel_model):
        with surfel_model.context.current():
            self.render_surfels_prg = tenviz.DrawProgram(
                tenviz.DrawMode.Points,
                vert_shader=_SHADER_DIR / "render_model.vert",
                frag_shader=_SHADER_DIR / "render_model.frag",
                #ignore_missing=True
            )

            self.render_surfels_prg['in_point'] = surfel_model.points
            self.render_surfels_prg['in_normal'] = surfel_model.normals
            self.render_surfels_prg['in_conf'] = surfel_model.confs
            self.render_surfels_prg['in_radius'] = surfel_model.radii
            self.render_surfels_prg['in_time'] = surfel_model.times
            self.render_surfels_prg['in_mask'] = surfel_model.active_mask_gl

            self.render_surfels_prg['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.render_surfels_prg['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.render_surfels_prg['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

            self.framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBFloat,  # pos
                1: tenviz.FramebufferTarget.RGBAFloat,  # normal + radius
                2: tenviz.FramebufferTarget.RGBInt32  # indexes + existence
            })

        self.surfel_model = surfel_model

    def raster(self, proj_matrix, rt_cam, width, height,
               stable_conf_thresh, time):
        context = self.surfel_model.context

        with context.current():
            self.render_surfels_prg['StableThresh'] = float(stable_conf_thresh)
            self.render_surfels_prg['Time'] = int(time)

        context.set_clear_color(0, 0, 0, 0)
        context.render(proj_matrix,
                       torch.from_numpy(rt_cam.opengl_view_cam).float(),
                       self.framebuffer,
                       [self.render_surfels_prg], width, height)

    @property
    def pos_tex(self):
        return self.framebuffer[0]

    @property
    def normal_rad_tex(self):
        return self.framebuffer[1]

    @property
    def idx_tex(self):
        return self.framebuffer[2]

    def show_debug(self, title, debug=True):
        if not debug:
            return

        with self.surfel_model.context.current():
            pos = self.pos_tex.to_tensor()
            normals = self.normal_rad_tex.to_tensor()
            idxs = self.idx_tex.to_tensor()

        plt.figure()
        plt.title("{} - pos".format(title))
        plt.imshow(pos[:, :, 2].cpu())

        plt.figure()
        plt.title("{} - normals".format(title))
        plt.imshow(normals.cpu())

        plt.figure()
        plt.title("{} - indices".format(title))
        plt.imshow(idxs[:, :, 0].cpu())

        plt.show()
