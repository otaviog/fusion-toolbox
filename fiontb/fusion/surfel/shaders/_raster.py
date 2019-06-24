from pathlib import Path

import tenviz

_SHADER_DIR = Path(__file__).parent / "shaders"


class GlobalSurfelRaster:
    def __init__(self, surfel_model):
        with surfel_model.context.current():
            self.render_surfels_prg = tenviz.DrawProgram(
                tenviz.DrawMode.Points,
                vert_shader=_SHADER_DIR / "render_model.vert",
                frag_shader=_SHADER_DIR / "render_model.frag")

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

    def raster(self, rt_cam, width, height, proj_matrix,
               stable_conf_thresh, time):
        context = self.surfel_model.context

        with context.current():
            self.render_surfels_prg['StableThresh'] = torch.tensor(
                [stable_conf_thresh], dtype=torch.float32)
            self.render_surfels_prg['Time'] = torch.tensor(
                [time], dtype=torch.int32)

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
