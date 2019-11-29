"""Raster for generating indexmaps.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tenviz

from fiontb.frame import Frame
from fiontb.camera import RTCamera, normal_transform_matrix
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

    def to_indexmap(self, device=None, non_blocking=False):
        with self.gl_context.current():
            indexmap = IndexMap()
            indexmap.position_confidence = self.framebuffer[0].to_tensor(
                non_blocking=non_blocking)
            indexmap.normal_radius = self.framebuffer[1].to_tensor(
                non_blocking=non_blocking)
            indexmap.color = self.framebuffer[2].to_tensor(
                non_blocking=non_blocking)
            indexmap.indexmap = self.framebuffer[3].to_tensor(
                non_blocking=non_blocking)

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
                vert_shader_file=_SHADER_DIR / "model_indexmap.vert",
                frag_shader_file=_SHADER_DIR / "indexmap.frag",
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

        super().__init__(surfel_model.gl_context)

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
            world_to_cam = rt_cam.world_to_cam
            self.program['WorldToCam'] = world_to_cam
            self.program['WorldToCamNormal'] = normal_transform_matrix(
                world_to_cam)
            self.program['StableThresh'] = (
                float(stable_conf_thresh)
                if stable_conf_thresh is not None else -1.0)

        if isinstance(rt_cam, RTCamera):
            view_cam = rt_cam.opengl_view_cam
        else:
            view_cam = rt_cam

        gl_context.clear_color = np.array([0, 0, 0, 0])
        gl_context.render(proj_matrix, view_cam,
                          self.framebuffer,
                          [self.program], width, height)


class SurfelIndexMapRaster(_BaseIndexMapRaster):
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
                vert_shader_file=_SHADER_DIR / "surfel_indexmap.vert",
                frag_shader_file=_SHADER_DIR / "surfel_indexmap.frag",
                geo_shader_file=_SHADER_DIR / "surfel_indexmap.geom",
                ignore_missing=True
            )

            self.program['in_point'] = surfel_model.positions
            self.program['in_normal'] = surfel_model.normals
            self.program['in_color'] = surfel_model.colors
            self.program['in_conf'] = surfel_model.confidences

            self.program['in_radius'] = surfel_model.radii
            self.program['in_mask'] = surfel_model.free_mask_gl
            self.program['in_time'] = surfel_model.times

            self.program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

        super().__init__(surfel_model.gl_context)

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

        if isinstance(rt_cam, RTCamera):
            view_cam = rt_cam.opengl_view_cam
            world_to_cam = rt_cam.world_to_cam
        else:
            view_cam = rt_cam
            world_to_cam = rt_cam

        with gl_context.current():
            self.program['WorldToCam'] = world_to_cam
            self.program['WorldToCamNormal'] = normal_transform_matrix(
                world_to_cam)
            self.program['StableThresh'] = (
                float(stable_conf_thresh)
                if stable_conf_thresh is not None else -1.0)


        gl_context.clear_color = np.array([0, 0, 0, 0])
        gl_context.render(proj_matrix, view_cam,
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
    plt.subplot(2, 3, 1)
    plt.imshow(indexmap.position_confidence[:, :, 0].cpu())
    plt.subplot(2, 3, 2)
    plt.imshow(indexmap.position_confidence[:, :, 1].cpu())
    plt.subplot(2, 3, 3)
    plt.imshow(indexmap.position_confidence[:, :, 2].cpu())

    plt.subplot(2, 3, 4)
    plt.imshow(indexmap.normal_radius.cpu())

    plt.subplot(2, 3, 5)
    plt.imshow(indexmap.indexmap[:, :, 0].cpu())

    plt.figure()
    plt.title("{} - Colors".format(title))
    plt.imshow(indexmap.color[:, :, :3].cpu())

    if show:
        plt.show()
