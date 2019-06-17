from pathlib import Path

import numpy as np
import torch

import tenviz

import matplotlib.pyplot as plt

_SHADER_DIR = Path(__file__).parent / "_indexmap"

IM_POS_FB = 2
IM_NORM_FB = 1
IM_IX_FB = 0
IM_DG_FB = 3

QU_IX_FB = 0
QU_DG_FB = 1


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
                IM_IX_FB: tenviz.FramebufferTarget.RGBInt32,
                IM_POS_FB: tenviz.FramebufferTarget.RGBFloat,
                IM_NORM_FB: tenviz.FramebufferTarget.RGBFloat,
                IM_DG_FB: tenviz.FramebufferTarget.RGBUint8
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
                QU_IX_FB: tenviz.FramebufferTarget.RGBInt32,
                QU_DG_FB: tenviz.FramebufferTarget.RGBFloat
            })

        self.indexmap_width = 0
        self.indexmap_height = 0

    def update(self, rt_cam, width, height, kcam, min_depth, max_depth):
        proj = tenviz.projection_from_kcam(
            kcam.matrix, min_depth, max_depth)
        proj_matrix = torch.from_numpy(proj.to_matrix()).float()
        view_mtx = rt_cam.opengl_view_cam

        self.indexmap_width = width
        self.indexmap_height = height

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix,
                            torch.from_numpy(view_mtx).float(),
                            self.update_fb,
                            [self.update_program], width, height)

    def get_visible_model_indices(self):
        attaches = self.update_fb.get_attachs()
        with self.context.current():
            model_idxs = attaches[IM_IX_FB].to_tensor()

        return model_idxs[model_idxs > 0].long()

    def query(self, surfel_cloud, width, height, kcam, min_depth, max_depth):
        view_mtx = np.eye(4)
        view_mtx[2, 2] = -1

        indexmap_out = self.update_fb.get_attachs()
        with self.context.current():
            self.query_program['IndexMapTex'] = indexmap_out[IM_IX_FB]

        with self.context.current():
            self.qpoint.from_tensor(surfel_cloud.points)
            self.qnormal.from_tensor(surfel_cloud.normals)

            self.query_program['ImageWidth'] = width
            self.query_program['ImageHeight'] = height
            self.query_program['Scale'] = self.indexmap_width/width

            self.query_program['IndexMapPointsTex'] = indexmap_out[IM_POS_FB]
            self.query_program['IndexMapNormalsTex'] = indexmap_out[IM_NORM_FB]

        proj = tenviz.projection_from_kcam(
            kcam.matrix, min_depth, max_depth)
        proj_matrix = torch.from_numpy(proj.to_matrix()).float()

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix,
                            torch.from_numpy(view_mtx).float(),
                            self.query_fb,
                            [self.query_program],
                            width, height)

        with self.context.current():
            buffers = self.query_fb.get_attachs()
            map_index = buffers[QU_IX_FB].to_tensor()
            # debug = buffers[QU_DG_FB].to_tensor().numpy()
            C = indexmap_out[IM_IX_FB].to_tensor().numpy()

        plt.figure()
        plt.title("query")
        plt.imshow(map_index[:, :, 1])
        plt.figure(); plt.title("indexmap"); plt.imshow(C[:, :, 0])
        # import ipdb; ipdb.set_trace()
        # plt.figure(); plt.imshow(map_index[:, :, 2])
        plt.show()

        write_mask = map_index[:, :, 0] != 0

        model_idxs = map_index[:, :, 1][write_mask]
        frame_idxs = map_index[:, :, 2][write_mask]

        fuse_mask = model_idxs > 0
        model_fuse_idxs = (model_idxs[fuse_mask] - 1).long()
        frame_fuse_idxs = frame_idxs[fuse_mask].long()
        frame_unstable_idxs = frame_idxs[model_idxs == 0].long()

        return (frame_fuse_idxs, model_fuse_idxs, frame_unstable_idxs)


def _main():
    import cv2
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud
    from fiontb.frame import Frame, FrameInfo, FramePointCloud

    test_data = Path(__file__).parent / "_indexmap"
    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))

    rt_cam = RTCamera.create_from_pos_quat(
        # 72
        0.401252, -0.0150952, 0.0846582, 0.976338, 0.0205504, -0.14868, -0.155681
    )

    indexmap = IndexMap(ctx, surfel_model)
    with ctx.current():
        idxs = torch.arange(0, model_size, dtype=torch.long)

        surfel_model.points[idxs] = model.verts
        surfel_model.normals[idxs] = model.normals
        surfel_model.colors[idxs] = model.colors
        surfel_model.radii[idxs] = torch.full(
            (model_size, ), 0.05, dtype=torch.float)
        surfel_model.confs[idxs] = torch.full(
            (model_size, ), 2.1, dtype=torch.float)

    indexmap.update(rt_cam, kcam, 0.01, 10.0)

    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    frame_pcl = FramePointCloud(frame)
    surfel_cloud = SurfelCloud.from_frame_pcl(frame_pcl)

    frame_idxs, model_idxs, frame_unstable_idxs = indexmap.query(
        surfel_cloud, 640, 480, kcam, 0.01, 10.0)


if __name__ == '__main__':
    _main()
