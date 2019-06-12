from pathlib import Path

import numpy as np
import torch

import tenviz


_SHADER_DIR = Path(__file__).parent / "_indexmap"


class IndexMap:
    # TODO set up frame buffer sizes
    def __init__(self, gl_context, surfel_model):
        self.context = gl_context
        with self.context.current():
            self.update_program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

            self.update_program['in_point'] = surfel_model.points
            self.update_program['in_normal'] = surfel_model.normals
            self.update_program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.update_program['Modelview'] = tenviz.MatPlaceholder.Modelview

            self.update_fb = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBFloat,
                1: tenviz.FramebufferTarget.RGBFloat,
                2: tenviz.FramebufferTarget.RInt32
            })

        with self.context.current():
            self.query_program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "query.vert",
                _SHADER_DIR / "query.frag", ignore_missing=True)

            self.qpoint = tenviz.buffer_create()
            self.qnormal = tenviz.buffer_create()

            self.query_program['in_point'] = self.qpoint
            self.query_program['in_normal'] = self.qnormal

            self.query_program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            self.query_fb = tenviz.create_framebuffer({
                0: tenviz.FramebufferTarget.RGBInt32,
                1: tenviz.FramebufferTarget.RGBFloat
            })
        self.context.set_clear_color(0, 255, 255, 0)

    def update(self, rt_cam, kcam, min_depth, max_depth):
        proj = tenviz.projection_from_kcam(
            kcam.matrix, min_depth, max_depth)
        proj_matrix = torch.from_numpy(proj.to_matrix()).float()

        view_mtx = rt_cam.opengl_view_cam
        self.context.render(proj_matrix,
                            torch.from_numpy(view_mtx).float(),
                            self.update_fb,
                            [self.update_program])

    def query(self, surfel_cloud, width, height, kcam, min_depth, max_depth):
        view_mtx = np.eye(4)
        view_mtx[2, 2] = -1

        with self.context.current():
            self.qpoint.from_tensor(surfel_cloud.points)
            self.qnormal.from_tensor(surfel_cloud.normals)

            indexmap_out = self.update_fb.get_attachs()
            self.query_program['IndexMapPointsTex'] = indexmap_out[0]
            self.query_program['IndexMapNormalsTex'] = indexmap_out[1]
            self.query_program['IndexMapTex'] = indexmap_out[2]

            self.query_program['ImageWidth'] = width
            self.query_program['ImageHeight'] = height
            self.query_program['Scale'] = 1.0

            # vc = tenviz.create_virtual_camera(
            # self.proj, np.linalg.inv(view_mtx))

        proj = tenviz.projection_from_kcam(
            kcam.matrix, min_depth, max_depth)
        proj_matrix = torch.from_numpy(proj.to_matrix()).float()

        self.context.render(proj_matrix,
                            torch.from_numpy(view_mtx).float(),
                            self.query_fb,
                            [self.query_program])

        buffers = self.query_fb.get_attachs()

        with self.context.current():
            map_index = buffers[0].to_tensor()

        valid_mask = map_index[:, :, 0] != 0
        import ipdb; ipdb.set_trace()
        frame_idxs = map_index[:, :, 2][valid_mask]
        model_idxs = map_index[:, :, 1][valid_mask]

        frame_stable_idxs = frame_idxs[model_idxs >= 0]
        frame_unstable_idxs = frame_idxs[model_idxs < 0]
        
        return frame_stable_idxs, model_idxs[model_idxs >= 0], frame_unstable_idxs


def _main():
    import cv2
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelsModel, SurfelCloud
    from fiontb.frame import Frame, FrameInfo, FramePointCloud

    test_data = Path(__file__).parent / "_indexmap"
    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelsModel(ctx, model_size*2)

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))

    rt_cam = RTCamera.create_from_pos_quat(
        # 72
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681
    )

    indexmap = IndexMap(ctx, surfel_model, kcam, 0.01, 10.0)
    with ctx.current():
        idxs = torch.arange(0, model_size, dtype=torch.long)

        surfel_model.points[idxs] = model.verts
        surfel_model.normals[idxs] = model.normals
        surfel_model.colors[idxs] = model.colors
        surfel_model.radii[idxs] = torch.full(
            (model_size, ), 0.05, dtype=torch.float)
        surfel_model.confs[idxs] = torch.full(
            (model_size, ), 2.1, dtype=torch.float)

    indexmap.update(rt_cam)

    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    frame_pcl = FramePointCloud(frame)
    surfel_cloud = SurfelCloud(frame_pcl)

    frame_idxs, model_idxs = indexmap.query(frame_pcl, surfel_cloud)


if __name__ == '__main__':
    _main()
