from pathlib import Path

import matplotlib.pyplot as plt

import tenviz
import torch

_SHADER_DIR = Path(__file__).parent


class IndexMap:
    def __init__(self, model_points, kcam, render_size, min_depth=0.01, max_depth=10.0):
        self.context = tenviz.Context()
        with self.context.current():
            self.model_program = tenviz.load_program_fs(
                _SHADER_DIR / "indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

            self.query_program = tenviz.load_program_fs(
                _SHADER_DIR / "indexmap_query.vert",
                _SHADER_DIR / "indexmap_query.frag")

            draw = tenviz.DrawProgram(
                tenviz.DrawMode.Points, program=self.program)

            draw['v_point'] = model_points
            index = torch.arange(0, model_points.size(
                0), dtype=torch.int32, device=model_points.device)
            draw['v_index'] = index

            self.proj = tenviz.projection_from_kcam(
                kcam.matrix, min_depth, max_depth)
            draw['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            draw.set_bounds(model_points)

            framebuffer = tenviz.create_framebuffer(
                {
                    0: tenviz.FramebufferTarget.RGBAUint8,
                    1: tenviz.FramebufferTarget.RInt32,
                    2: tenviz.FramebufferTarget.RGBFloat
                })

        self.context.set_clear_color(0, -1, 255, -1)
        #viewer = self.context.viewer([draw], tenviz.CameraManipulator.WASD)
        # viewer.reset_view()
        while False:
            key = viewer.wait_key(1)
            if key < 0:
                break

        self.context.render(
            torch.eye(4).float(),
            torch.from_numpy(self.proj.to_matrix()).float(), framebuffer,
            [draw])

        buffers = framebuffer.get_attachs()

        self.model_index_tex = buffers[1]
        self.model_point_tex = buffers[2]

        with self.context.current():
            indexmap = self.model_index_tex.to_tensor()

        # plt.imshow(indexmap[:, :].byte())
        # plt.show()

    def query(self, query_points, max_k, radius):
        with self.context.current():
            draw = tenviz.DrawProgram(
                tenviz.DrawMode.Points, program=self.query_program)
            draw['v_qpoint'] = query_points
            draw['v_qindex'] = torch.arange(0, query_points.size(
                0), dtype=torch.int32, device=query_points.device)

            draw['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            draw['IndexSampler'] = self.model_index_tex
            draw['ModelPointSampler'] = self.model_point_tex
            # draw['IndexXSteps'] = 4
            # draw['IndexYSteps'] = 4

            draw.set_bounds(query_points)

            framebuffer = tenviz.create_framebuffer(
                {
                    0: tenviz.FramebufferTarget.RGBAUint8,
                    1: tenviz.FramebufferTarget.RInt32,
                    2: tenviz.FramebufferTarget.RInt32
                })

        self.context.render(
            torch.eye(4).float(),
            torch.from_numpy(self.proj.to_matrix()).float(),
            framebuffer, [draw])

        if False:
            viewer = self.context.viewer([draw], tenviz.CameraManipulator.WASD)
            viewer.reset_view()
            while True:
                key = viewer.wait_key(1)
                if key < 0:
                    break

        buffers = framebuffer.get_attachs()
        with self.context.current():
            image = buffers[0].to_tensor()
            query_index = buffers[1].to_tensor()
            model_index = buffers[2].to_tensor()

        plt.figure()
        plt.imshow(image[:, :, :3].byte())
        plt.figure()
        plt.imshow(query_index[:, :])
        plt.figure()
        plt.imshow(model_index[:, :])
        plt.show()

        query_mask = query_index > 0
        query_index = query_index[query_mask]
        model_index = model_index[query_mask]
