from enum import Enum

import torch
import cv2

import tenviz

from fiontb.viz.surfelrender import SurfelRender, RenderMode
from fiontb.spatial.trigoctree import TrigOctree


class RunMode(Enum):
    PLAY = 0
    STEP = 1


class SurfelReconstructionUI:
    def __init__(self, surfel_model, run_mode=RunMode.PLAY, inverse=False, gt_mesh=None):
        self.run_mode = run_mode

        self.surfel_render = SurfelRender(surfel_model)

        scene = [self.surfel_render]
        if gt_mesh is not None:
            with surfel_model.gl_context.current():
                self.gt_mesh_node = tenviz.create_mesh(
                    gt_mesh.verts, gt_mesh.faces, normals=gt_mesh.normals)
                scene.append(self.gt_mesh_node)

            self.oct_tree = TrigOctree(
                gt_mesh.verts, gt_mesh.faces.long(), 256)
        self.gt_mesh = gt_mesh

        self.viewer = surfel_model.gl_context.viewer(
            scene, cam_manip=tenviz.CameraManipulator.WASD)
        self.viewer.reset_view()
        self.surfel_model = surfel_model

        if inverse:
            inv = torch.eye(4, dtype=torch.float32)
            inv[1, 1] = -1
            inv[0, 0] = -1
            self.surfel_render.set_transform(inv)

    def __iter__(self):
        frame_count = 0
        quit_flag = False
        read_next_frame = True
        use_camera_view = False

        while not quit_flag:
            if read_next_frame:
                print("Next frame: {}".format(frame_count))
                yield True
                read_next_frame = self.run_mode != RunMode.STEP
                frame_count += 1

            self.surfel_model.gl_context.set_clear_color(0.32, 0.34, 0.87, 1)
            keys = [self.viewer.wait_key(0), cv2.waitKey(1)]

            for key in keys:
                key = key & 0xff
                if key == 27:
                    quit_flag = True

                key = chr(key).lower()

                if key == 'm':
                    if self.run_mode == RunMode.PLAY:
                        self.run_mode = RunMode.STEP
                        read_next_frame = False
                    else:
                        self.run_mode = RunMode.PLAY
                        read_next_frame = True
                elif key == 'q':
                    quit_flag = True
                elif key == 'n':
                    read_next_frame = True
                elif key == 'i':
                    self.surfel_render.set_render_mode(RenderMode.Confs)
                    self.surfel_render.set_max_confidence(1024)
                elif key == 'u':
                    self.surfel_render.set_render_mode(RenderMode.Color)
                elif key == 'o':
                    self.surfel_render.set_render_mode(RenderMode.Normal)
                elif key == 'y':
                    self.surfel_render.set_render_mode(RenderMode.Times)
                    self.surfel_render.set_max_time(self.surfel_model.max_time)
                elif key == 'p':
                    self.surfel_render.set_stable_threshold(10)
                elif key == 'l':
                    self.surfel_render.set_stable_threshold(-1)
                elif key == 't':
                    self.surfel_render.set_render_mode(RenderMode.Ids)
                elif key == 'c':
                    use_camera_view = not use_camera_view
                elif key == 'v':
                    self._eval_accuracy()
                elif key == 'b':
                    import ipdb
                    ipdb.set_trace()

            if self.surfel_render.render_mode == RenderMode.Confs:
                with self.surfel_model.context.current():
                    with self.surfel_model.confs.as_tensor() as confs:
                        max_conf = confs[self.surfel_model.active_mask].max()

        cv2.destroyAllWindows()

    def _eval_accuracy(self):
        if self.gt_mesh is None:
            return

        rec_verts = self.fusion_ctx.get_stable_points().cpu()
        gt_closest, _ = self.oct_tree.query_closest_points(rec_verts)

        mesh_acc = mesh_accuracy(rec_verts, gt_closest)
        print("Current accuracy: {}".format(mesh_acc))
