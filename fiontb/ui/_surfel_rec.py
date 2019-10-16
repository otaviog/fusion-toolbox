from enum import Enum
import ipdb

import torch
import cv2

import tenviz
import tenviz.io

from fiontb.viz.surfelrender import SurfelRender, RenderMode
from fiontb.spatial.trigoctree import TrigOctree


class RunMode(Enum):
    PLAY = 0
    STEP = 1


class SurfelReconstructionUI:
    def __init__(self, surfel_model, run_mode=RunMode.PLAY, inverse=False, gt_mesh=None,
                 stable_conf_thresh=10.0, time_thresh=20):
        self.run_mode = run_mode
        self.surfel_render = SurfelRender(surfel_model)

        self.stable_conf_thresh = stable_conf_thresh
        self.time_thresh = time_thresh

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

        self._quit_flag = False
        self._read_next_frame = True
        self._use_camera_view = False
        self.frame_count = 0

    def next_frame(self):
        self._read_next_frame = True

    def quit(self):
        self._quit_flag = True

    def toggle_run_mode(self):
        if self.run_mode == RunMode.PLAY:
            self.run_mode = RunMode.STEP
            self._read_next_frame = False
            print("Run mode set to step")
        else:
            self.run_mode = RunMode.PLAY
            self._read_next_frame = True
            print("Run mode set to continuous")

    def set_render_mode(self, mode):
        self.surfel_render.set_render_mode(mode)

    def show_stable_only(self):
        print("Show only stable surfels")
        self.surfel_render.set_stable_threshold(self.stable_conf_thresh)

    def show_unstable_too(self):
        print("Show unstable and stable")
        self.surfel_render.set_stable_threshold(-1)

    def toggle_camera_mode(self):
        self._use_camera_view = not self._use_camera_view

    def save_model(self, output_filename):
        cloud = self.surfel_model.to_surfel_cloud()[0]
        good_confs = cloud.confidences > self.stable_conf_thresh
        cloud = cloud[good_confs]
        tenviz.io.write_3dobject(
            output_filename, cloud.positions.cpu().numpy(),
            normals=cloud.normals,
            colors=cloud.colors)

    def __iter__(self):
        self.frame_count = 0
        use_camera_view = False

        key_map = {'q': self.quit,
                   'n': self.next_frame,
                   'm': self.toggle_run_mode,
                   'r': lambda: self.set_render_mode(RenderMode.Color),
                   't': lambda: self.set_render_mode(RenderMode.Normal),
                   'y': lambda: self.set_render_mode(RenderMode.Confs),
                   'u': lambda: self.set_render_mode(RenderMode.Times),
                   'i': lambda: self.set_render_mode(RenderMode.Ids),
                   'o': lambda: self.set_render_mode(RenderMode.Gray),
                   'f': self.show_stable_only,
                   'g': self.show_unstable_too,
                   'h': self.toggle_camera_mode,
                   'j': self._eval_accuracy,
                   'v': lambda: self.save_model("model-{}.ply".format(
                       self.surfel_model.max_time))
                   }
        print("""Key controls:
        q: quit
        n: next frame
        m: toggle between step and continuous play mode
        r: render colors
        t: render normals
        y: render confidences
        u: render times
        i: render surfel ids
        o: render gray scale
        f: show stable only
        g: show unstable too
        h: toggle between camera modes
        j: evaluate accuracy
        """)

        while not self._quit_flag:
            if self._read_next_frame:
                yield True
                self._read_next_frame = self.run_mode != RunMode.STEP
                self.frame_count += 1

                with self.surfel_model.gl_context.current():
                    self.surfel_render.set_max_confidence(
                        self.surfel_model.max_confidence)
                    self.surfel_render.set_max_time(self.surfel_model.max_time)

            self.surfel_model.gl_context.set_clear_color(0.32, 0.34, 0.87, 1)
            keys = [self.viewer.wait_key(0), cv2.waitKey(1)]

            if keys[0] < 0:
                self.quit()

            for key in keys:
                if key & 0xff == 27:
                    self.quit()
                    break
                key = chr(key & 0xff).lower()
                if key == 'z':
                    ipdb.set_trace()

                if key in key_map:
                    key_map[key]()

        cv2.destroyAllWindows()

    def _eval_accuracy(self):
        if self.gt_mesh is None:
            return

        rec_verts = self.fusion_ctx.get_stable_points().cpu()
        gt_closest, _ = self.oct_tree.query_closest_points(rec_verts)

        mesh_acc = mesh_accuracy(rec_verts, gt_closest)
        print("Current accuracy: {}".format(mesh_acc))
