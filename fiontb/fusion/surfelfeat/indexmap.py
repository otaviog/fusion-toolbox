"""Raster for generating indexmaps.
"""

from pathlib import Path

import torch
import matplotlib.pyplot as plt

import tenviz

from fiontb.frame import Frame
from fiontb._cfiontb import IndexMap

_SHADER_DIR = Path(__file__).parent / "shaders"


class _BaseIndexMapRaster:
    def __init__(self, context):
        self.context = context

        with context.current():
            self.framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBAFloat,
                1: tenviz.FramebufferTarget.RGBAFloat,
                2: tenviz.FramebufferTarget.RGBUint8,
                3: tenviz.FramebufferTarget.RGBInt32
            })

    def get(self, device=None):
        indexmap = IndexMap()
        indexmap.position_confidence = self.framebuffer[0].to_tensor()
        indexmap.normal_radius = self.framebuffer[1].to_tensor()
        indexmap.color = self.framebuffer[2].to_tensor()
        indexmap.indexmap = self.framebuffer[3].to_tensor()

        if device is not None:
            indexmap = indexmap.to(str(device))

        return indexmap

    def to_frame(self, frame_info):
        indexmap = self.get()

        depth = indexmap.position_confidence[:, :, 2]
        color = indexmap.color.to_tensor()

        depth = (depth*(1.0 / frame_info.depth_scale)
                 + frame_info.depth_bias).round().int().cpu().numpy()
        color = color.cpu().numpy()

        return Frame(frame_info, depth, color)


class LiveIndexMapRaster(_BaseIndexMapRaster):
    """Rasterizer of :obj:`fiontb.fusion.surfel.model.SurfelCloud` that
    writes position, normal and the the surfels' index to
    framebuffers.
    """

    def __init__(self, context):
        with context.current():
            self.program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "live_indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

            self.positions = tenviz.buffer_create()
            self.confidences = tenviz.buffer_create()
            self.normals = tenviz.buffer_create()
            self.radii = tenviz.buffer_create()
            self.colors = tenviz.buffer_create()

            self.program['in_point'] = self.positions
            self.program['in_conf'] = self.confidences
            self.program['in_normal'] = self.normals
            self.program['in_radius'] = self.radii
            self.program['in_color'] = self.colors

            self.program['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.program['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview
            self.program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        super(LiveIndexMapRaster, self).__init__(context)

    def raster(self, live_surfels, proj_matrix, width, height):
        """Raster the surfelcloud to the framebuffers.

        Args:

            live_surfels
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
            self.positions.from_tensor(live_surfels.positions)
            self.confidences.from_tensor(live_surfels.confidences)
            self.normals.from_tensor(live_surfels.normals)
            self.radii.from_tensor(live_surfels.radii)
            self.colors.from_tensor(live_surfels.colors)

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix, view_mtx, self.framebuffer,
                            [self.program], width, height)


class ModelIndexMapRaster(_BaseIndexMapRaster):
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

            self.render_surfels_prg['in_point'] = surfel_model.positions
            self.render_surfels_prg['in_normal'] = surfel_model.normals
            self.render_surfels_prg['in_color'] = surfel_model.colors
            self.render_surfels_prg['in_conf'] = surfel_model.confidences
            self.render_surfels_prg['in_radius'] = surfel_model.radii
            self.render_surfels_prg['in_mask'] = surfel_model.active_mask_gl

            self.render_surfels_prg['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.render_surfels_prg['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.render_surfels_prg['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

        super(ModelIndexMapRaster, self).__init__(surfel_model.context)

        self.surfel_model = surfel_model

    def raster(self, proj_matrix, rt_cam, width, height,
               stable_conf_thresh=None):
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

        context.set_clear_color(0, 0, 0, 0)
        context.render(proj_matrix, rt_cam.opengl_view_cam,
                       self.framebuffer,
                       [self.render_surfels_prg], width, height)


def show_indexmap(indexmap, title, debug=True):
    """Helper function for quickly display the framebuffers in pyplot.

    Args:

        title (str): The plot's base title.

        debug (bool): Are you really serious about debugging?
    """
    if not debug:
        return

    plt.figure()
    plt.title("{} - Positions".format(title))
    plt.imshow(indexmap.position_confidence[:, :, 2].cpu())

    plt.figure()
    plt.title("{} - Normals".format(title))
    plt.imshow(indexmap.normal_radius.cpu())

    plt.figure()
    plt.title("{} - Indices".format(title))
    plt.imshow(indexmap.indexmap[:, :, 0].cpu())

    plt.show()
