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
    def __init__(self, gl_context):
        self.gl_context = gl_context

        with gl_context.current():
            self.framebuffer = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBAFloat,
                1: tenviz.FramebufferTarget.RGBAFloat,
                2: tenviz.FramebufferTarget.RGBUint8,
                3: tenviz.FramebufferTarget.RGBInt32
            })

    def to_indexmap(self, device=None):
        with self.gl_context.current():
            indexmap = IndexMap()
            indexmap.position_confidence = self.framebuffer[0].to_tensor(
                non_blocking=True)
            indexmap.normal_radius = self.framebuffer[1].to_tensor(
                non_blocking=True)
            indexmap.color = self.framebuffer[2].to_tensor(non_blocking=True)
            indexmap.indexmap = self.framebuffer[3].to_tensor(
                non_blocking=True)

        if device is not None:
            indexmap = indexmap.to(str(device))

        return indexmap

    def to_frame(self, frame_info):
        indexmap = self.to_indexmap()

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

    def __init__(self, gl_context):
        with gl_context.current():
            self.program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "live_indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

            self.positions = tenviz.buffer_create()
            self.confidences = tenviz.buffer_create()
            self.normals = tenviz.buffer_create()
            self.radii = tenviz.buffer_create()
            self.colors = tenviz.buffer_create(normalize=True)
            self.times = tenviz.buffer_create()

            self.program['in_point'] = self.positions
            self.program['in_conf'] = self.confidences
            self.program['in_normal'] = self.normals
            self.program['in_radius'] = self.radii
            self.program['in_color'] = self.colors
            self.program['in_time'] = self.times

            self.program['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.program['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview
            self.program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        super(LiveIndexMapRaster, self).__init__(gl_context)

    _VIEW_MTX = torch.eye(4, dtype=torch.float32)
    _VIEW_MTX[2, 2] = -1

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

        with self.gl_context.current():
            self.positions.from_tensor(live_surfels.positions)
            self.confidences.from_tensor(live_surfels.confidences)
            self.normals.from_tensor(live_surfels.normals)
            self.radii.from_tensor(live_surfels.radii)
            self.colors.from_tensor(live_surfels.colors)
            self.times.from_tensor(live_surfels.times)

        self.gl_context.set_clear_color(0, 0, 0, 0)
        self.gl_context.render(proj_matrix, LiveIndexMapRaster._VIEW_MTX, self.framebuffer,
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

        with surfel_model.gl_context.current():
            self.program = tenviz.DrawProgram(
                tenviz.DrawMode.Points,
                vert_shader=_SHADER_DIR / "model_indexmap.vert",
                frag_shader=_SHADER_DIR / "indexmap.frag",
                # ignore_missing=True
            )

            self.program['in_point'] = surfel_model.positions
            self.program['in_normal'] = surfel_model.normals
            self.program['in_color'] = surfel_model.colors
            self.program['in_conf'] = surfel_model.confidences
            self.program['in_radius'] = surfel_model.radii
            self.program['in_mask'] = surfel_model.free_mask_gl
            self.program['in_time'] = surfel_model.times

            self.program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.program['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.program['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

        super(ModelIndexMapRaster, self).__init__(surfel_model.gl_context)

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
        gl_context = self.surfel_model.gl_context

        with gl_context.current():
            self.program['StableThresh'] = (
                float(stable_conf_thresh)
                if stable_conf_thresh is not None else -1.0)

        gl_context.set_clear_color(0, 0, 0, 0)
        gl_context.render(proj_matrix, rt_cam.opengl_view_cam,
                          self.framebuffer,
                          [self.program], width, height)


def show_indexmap(indexmap, title='', debug=True, show=True):
    """Helper function for quickly display the framebuffers in pyplot.

    Args:

        title (str): The plot's base title.

        debug (bool): Are you really serious about debugging?
    """
    if not debug:
        return

    plt.figure()
    plt.title("{} - Positions".format(title))
    plt.subplot(1, 3, 1)
    plt.imshow(indexmap.position_confidence[:, :, 0].cpu())
    plt.subplot(1, 3, 2)
    plt.imshow(indexmap.position_confidence[:, :, 1].cpu())
    plt.subplot(1, 3, 3)
    plt.imshow(indexmap.position_confidence[:, :, 2].cpu())

    plt.figure()
    plt.title("{} - Normals".format(title))
    plt.imshow(indexmap.normal_radius.cpu())

    plt.figure()
    plt.title("{} - Indices".format(title))
    plt.imshow(indexmap.indexmap[:, :, 0].cpu())

    plt.figure()
    plt.title("{} - Colors".format(title))
    plt.imshow(indexmap.color[:, :, :3].cpu())

    if show:
        plt.show()
