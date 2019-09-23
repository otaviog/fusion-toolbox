"""Raster for generating indexmaps.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt

import tenviz

from fiontb.frame import Frame

_SHADER_DIR = Path(__file__).parent / "shaders"


class LiveIndexMap:
    """Rasterizer of :obj:`fiontb.fusion.surfel.model.SurfelCloud` that
    writes position, normal and the the surfels' index to
    framebuffers.
    """

    def __init__(self, context):

        with context.current():
            self.program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "live_indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

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
        """Raster the surfelcloud to the framebuffers.

        Args:

            surfel_cloud
             (:obj:`fiontb.fusion.surfel.model.SurfelCloud`): Camera's
             surfel cloud.

            proj_matrix (:obj:`torch.Tensor`): OpenGL style projection
             matrix reproducing the real camera one.

            width (int): Image width.

            height (int): Image height.

        """

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
        """Texture from the last raster position framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGB float texture
        """
        return self.framebuffer[0]

    @property
    def normal_rad_tex(self):
        """Texture from the last raster normal and radius framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGBA float texture, `A` is
        the radius value.
        """

        return self.framebuffer[1]

    @property
    def idx_tex(self):
        """Texture from the last raster index framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGB int32 texture, `R` is
        the surfel index, and `B` is equal 1 if it's a valid position,
        0 otherwise.

        """
        return self.framebuffer[2]

    def show_debug(self, title, debug=True):
        """Helper function for quickly display the framebuffers in pyplot.

        Args:

            title (str): The plot's base title.

            debug (bool): Are you really serious about debugging?
        """
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


class ModelIndexMap:
    """Rasterizer of :obj:`fiontb.fusion.surfel.model.SurfelModel` that
    writes position, normal and the the surfels' index to
    framebuffers.
    """

    def __init__(self, surfel_model):
        """Args:

            surfel_model
             (:obj:`fiontb.fusion.surfel.model.SurfelModel`): Surfel model.

        """

        with surfel_model.context.current():
            self.render_surfels_prg = tenviz.DrawProgram(
                tenviz.DrawMode.Points,
                vert_shader=_SHADER_DIR / "model_indexmap.vert",
                frag_shader=_SHADER_DIR / "indexmap.frag",
                # ignore_missing=True
            )

            self.render_surfels_prg['in_point'] = surfel_model.points
            self.render_surfels_prg['in_normal'] = surfel_model.normals
            self.render_surfels_prg['in_color'] = surfel_model.colors
            self.render_surfels_prg['in_conf'] = surfel_model.confs
            self.render_surfels_prg['in_radius'] = surfel_model.radii
            self.render_surfels_prg['in_time'] = surfel_model.times
            self.render_surfels_prg['in_mask'] = surfel_model.active_mask_gl

            self.render_surfels_prg['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.render_surfels_prg['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.render_surfels_prg['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

            self.framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBAFloat,  # pos + conf
                1: tenviz.FramebufferTarget.RGBAFloat,  # normal + radius
                2: tenviz.FramebufferTarget.RGBInt32,  # index + existence
                3: tenviz.FramebufferTarget.RGBUint8  # debug
            })

        self.surfel_model = surfel_model
        self.is_rasterized = False

    @property
    def context(self):
        return self.surfel_model.context

    def raster(self, proj_matrix, rt_cam, width, height,
               stable_conf_thresh=None, time=None):
        """Raster the surfel model to the framebuffers.

        Args:

            proj_matrix (:obj:`torch.Tensor`): OpenGL style projection
             matrix reproducing the real camera one.

            rt_cam (:obj:`fiontb.camera.RTCamera`): Viewpoint camera transformation.

            width (int): Image width.

            height (int): Image height.

        """

        context = self.surfel_model.context

        with context.current():
            self.render_surfels_prg['StableThresh'] = (
                float(stable_conf_thresh)
                if stable_conf_thresh is not None else -1.0)
            self.render_surfels_prg['Time'] = (
                int(time) if time is not None else -1)

        context.set_clear_color(0, 0, 0, 0)
        context.render(proj_matrix, rt_cam.opengl_view_cam,
                       self.framebuffer,
                       [self.render_surfels_prg], width, height)
        self.is_rasterized = True

    @property
    def position_confidence_tex(self):
        """Texture from the last raster position framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGB float texture
        """

        return self.framebuffer[0]

    @property
    def normal_radius_tex(self):
        """Texture from the last raster normal and radius framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGBA float texture, `A` is
        the radius value.
        """

        return self.framebuffer[1]

    @property
    def index_tex(self):
        """Texture from the last raster index framebuffer.

        Returns: (:obj:`tenviz.Texture`): RGB int32 texture, `R` is
        the surfel index, and `B` is equal 1 if it's a valid position,
        0 otherwise.

        """

        return self.framebuffer[2]

    @property
    def color_tex(self):
        return self.framebuffer[3]

    def to_params(self):
        params = IndexMapParams()
        params.position_confidence = self.position_confidence_tex.to_tensor()
        params.normal_radius = self.normal_radius_tex.to_tensor()
        params.color = self.color_tex.to_tensor()
        params.indexmap = self.index_tex.to_tensor()
        return params

    def to_frame(self, frame_info):
        with self.surfel_model.context.current():
            depth = self.position_confidence_tex.to_tensor()[:, :, 2]
            color = self.color_tex.to_tensor()

        depth = (depth*(1.0 / frame_info.depth_scale)
                 + frame_info.depth_bias).round().int().cpu().numpy()
        color = color.cpu().numpy()

        return Frame(frame_info, depth, color)

    def show_debug(self, title, debug=True):
        """Helper function for quickly display the framebuffers in pyplot.

        Args:

            title (str): The plot's base title.

            debug (bool): Are you really serious about debugging?
        """

        if not debug:
            return

        with self.surfel_model.context.current():
            pos = self.position_confidence_tex.to_tensor()
            normals = self.normal_radius_tex.to_tensor()
            idxs = self.index_tex.to_tensor()

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
