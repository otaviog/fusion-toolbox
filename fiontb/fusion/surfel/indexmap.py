from pathlib import Path

import numpy as np
import torch

import tenviz

import matplotlib.pyplot as plt

_SHADER_DIR = Path(__file__).parent / "shaders"

_FB_UPDT_POS = 2
_FB_UPDT_NORM = 1
_FB_UPDT_IDX = 0
_FB_UPDT_DBG = 3

_FB_QUE_IDX = 0
_FB_QUE_DBG = 1


class IndexMap:
    def __init__(self, gl_context, surfel_model):
        self.context = gl_context
        self.surfel_model = surfel_model
        with self.context.current():
            self.update_program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "indexmap.vert",
                _SHADER_DIR / "indexmap.frag")

            self.update_program['in_point'] = surfel_model.points
            self.update_program['in_normal'] = surfel_model.normals
            self.update_program['in_mask'] = surfel_model.active_mask_gl

            self.update_program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview
            self.update_program['Modelview'] = tenviz.MatPlaceholder.Modelview
            self.update_program['NormalModelview'] = tenviz.MatPlaceholder.NormalModelview

            self.update_fb = tenviz.create_framebuffer({
                _FB_UPDT_IDX: tenviz.FramebufferTarget.RGBInt32,
                _FB_UPDT_POS: tenviz.FramebufferTarget.RGBFloat,
                _FB_UPDT_NORM: tenviz.FramebufferTarget.RGBFloat,
                _FB_UPDT_DBG: tenviz.FramebufferTarget.RGBUint8
            })

        with self.context.current():
            self.query_program = tenviz.DrawProgram(
                tenviz.DrawMode.Points, _SHADER_DIR / "query.vert",
                _SHADER_DIR / "query.frag")

            self.qpoint = tenviz.buffer_create()
            self.qnormal = tenviz.buffer_create()

            self.query_program['in_point'] = self.qpoint
            self.query_program['in_normal'] = self.qnormal

            self.query_program['ProjModelview'] = tenviz.MatPlaceholder.ProjectionModelview

            self.query_fb = tenviz.create_framebuffer({
                _FB_QUE_IDX: tenviz.FramebufferTarget.RGBInt32,
                _FB_QUE_DBG: tenviz.FramebufferTarget.RGBFloat
            })

        self.indexmap_width = 0
        self.indexmap_height = 0

    def update(self, rt_cam, width, height, proj_matrix, debug=False):
        self.indexmap_width = width
        self.indexmap_height = height

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix,
                            torch.from_numpy(rt_cam.opengl_view_cam).float(),
                            self.update_fb,
                            [self.update_program], width, height)

        if debug:
            with self.context.current():
                plt.figure()
                plt.title("IndexMap")
                plt.imshow(
                    self.update_fb[_FB_UPDT_IDX].to_tensor().cpu()[:, :, 0])
                plt.show()

    def get_visible_model_indices(self):
        with self.context.current():
            model_idxs = self.update_fb[_FB_UPDT_IDX].to_tensor()

        return model_idxs[model_idxs > 0].long()

    def query(self, surfel_cloud, width, height, proj_matrix,
              max_normal_angle=0.5, debug=False):
        view_mtx = np.eye(4)
        view_mtx[2, 2] = -1

        with self.context.current():
            self.qpoint.from_tensor(surfel_cloud.points)
            self.qnormal.from_tensor(surfel_cloud.normals)

            self.query_program['ImageWidth'] = width
            self.query_program['ImageHeight'] = height
            self.query_program['Scale'] = self.indexmap_width/width

            self.query_program['IndexMapTex'] = self.update_fb[_FB_UPDT_IDX]
            self.query_program['IndexMapPointsTex'] = self.update_fb[_FB_UPDT_POS]
            self.query_program['IndexMapNormalsTex'] = self.update_fb[_FB_UPDT_NORM]
            self.query_program['MaxNormalAngle'] = max_normal_angle

        self.context.set_clear_color(0, 0, 0, 0)
        self.context.render(proj_matrix,
                            torch.from_numpy(view_mtx).float(),
                            self.query_fb,
                            [self.query_program],
                            width, height)

        with self.context.current():
            map_index = self.query_fb[_FB_QUE_IDX].to_tensor()

        if debug:
            plt.figure()
            plt.title("query")
            plt.imshow(map_index[:, :, 1].cpu())
            plt.figure()
            with self.context.current():
                indexmap = self.update_fb[_FB_UPDT_IDX].to_tensor()
            plt.title("indexmap")
            plt.imshow(indexmap.cpu()[:, :, 0])
            plt.show()

        write_mask = map_index[:, :, 0] != 0

        model_idxs = map_index[:, :, 1][write_mask]
        frame_idxs = map_index[:, :, 2][write_mask]

        fuse_mask = model_idxs > 0
        model_fuse_idxs = (model_idxs[fuse_mask]).long()
        frame_fuse_idxs = frame_idxs[fuse_mask].long()
        frame_unstable_idxs = frame_idxs[model_idxs == 0].long()

        return (frame_fuse_idxs, model_fuse_idxs, frame_unstable_idxs)


def _test():
    import cv2
    import tenviz.io

    from fiontb.camera import KCamera, RTCamera
    from fiontb.fusion.surfel import SurfelModel, SurfelCloud
    from fiontb.frame import Frame, FrameInfo, FramePointCloud

    test_data = Path(__file__).parent / "_test"
    model = tenviz.io.read_3dobject(test_data / "chair-model.ply").torch()
    model_size = model.verts.size(0)

    ctx = tenviz.Context()
    surfel_model = SurfelModel(ctx, model_size*2)

    kcam = KCamera.create_from_params(
        481.20001220703125, 480.0, (319.5, 239.5))
    proj = tenviz.projection_from_kcam(
        kcam.matrix, 0.01, 10.0)
    proj_matrix = torch.from_numpy(proj.to_matrix()).float()

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
        surfel_model.active_mask[idxs] = torch.zeros(
            (model_size,), dtype=torch.uint8, device="cuda:0")

    surfel_model.update_active_mask_gl()

    indexmap.update(rt_cam, 640, 480, proj_matrix, debug=True)

    frame_depth = cv2.imread(
        str(test_data / "chair-next-depth.png"), cv2.IMREAD_ANYDEPTH)
    frame_color = cv2.cvtColor(cv2.imread(
        str(test_data / "chair-next-rgb.png")), cv2.COLOR_BGR2RGB)

    frame = Frame(FrameInfo(kcam, depth_scale=1.0/5000),
                  frame_depth, frame_color)
    frame_pcl = FramePointCloud(frame)
    surfel_cloud = SurfelCloud.from_frame_pcl(frame_pcl, 0, device="cuda:0")

    frame_idxs, model_idxs, frame_unstable_idxs = indexmap.query(
        surfel_cloud, 640, 480, proj_matrix, debug=True)


if __name__ == '__main__':
    _test()
