"""
"""

from enum import Enum

import numpy as np
from matplotlib.pyplot import get_cmap
import cv2
import torch

import tenviz

from fiontb.frame import FramePointCloud, estimate_normals
from fiontb.filtering import bilateral_filter_depth_image
from fiontb.viz.surfelrender import SurfelRender, RenderMode
from fiontb.ui import convert_normals_to_rgb
from fiontb.metrics import mesh_accuracy
from fiontb.spatial.trigoctree import TrigOctree


class RunMode(Enum):
    PLAY = 0
    STEP = 1


class SensorFrameUI:
    _DEPTH_OPPACITY_LABEL = "depth oppacity"
    _NORMAL_OPPACITY_LABEL = "normal oppacity"

    def __init__(self, title):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(SensorFrameUI._DEPTH_OPPACITY_LABEL,
                           self.title, 50, 100,
                           self._update)
        cv2.createTrackbar(SensorFrameUI._NORMAL_OPPACITY_LABEL,
                           self.title, 50, 100,
                           self._update)

        self.frame = None
        self.normal_image = None

    def _update(self, _):
        if self.frame is None:
            return

        cmap = get_cmap('plasma', self.frame.info.depth_max)
        depth_img = (self.frame.depth_image / self.frame.info.depth_max)
        depth_img = cmap(depth_img)
        depth_img = depth_img[:, :, 0:3]
        depth_img = (depth_img*255).astype(np.uint8)

        depth_alpha = cv2.getTrackbarPos(
            SensorFrameUI._DEPTH_OPPACITY_LABEL, self.title) / 100.0

        canvas = cv2.addWeighted(depth_img, depth_alpha,
                                 self.frame.rgb_image, 1.0 - depth_alpha, 0.0)

        if self.normal_image is not None:
            normal_alpha = cv2.getTrackbarPos(
                SensorFrameUI._NORMAL_OPPACITY_LABEL, self.title) / 100.0

            canvas = cv2.addWeighted(self.normal_image, normal_alpha,
                                     canvas, 1.0 - normal_alpha, 0.0)

        cv2.imshow(self.title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def update(self, frame):
        self.frame = frame
        if frame.normal_image is not None:
            self.normal_image = convert_normals_to_rgb(frame.normal_image.cpu())
        self._update(0)


class MainLoop:
    def __init__(self, sensor, surfel_model, rec_step, fusion_ctx,
                 max_frames=None, run_mode=RunMode.PLAY, gt_mesh=None):
        self.sensor = sensor
        self.rec_step = rec_step
        self.fusion_ctx = fusion_ctx

        self.max_frames = max_frames
        self.run_mode = run_mode

        self.surfel_render = SurfelRender(surfel_model)

        scene = [self.surfel_render]
        if gt_mesh is not None:
            with surfel_model.context.current():
                self.gt_mesh_node = tenviz.create_mesh(
                    gt_mesh.verts, gt_mesh.faces, normals=gt_mesh.normals)
                scene.append(self.gt_mesh_node)

            self.oct_tree = TrigOctree(
                gt_mesh.verts, gt_mesh.faces.long(), 256)
        self.gt_mesh = gt_mesh

        self.viewer = surfel_model.context.viewer(
            scene, cam_manip=tenviz.CameraManipulator.WASD)
        self.viewer.reset_view()
        self.surfel_model = surfel_model

        # inv_y = np.eye(4, dtype=np.float32)
        # inv_y[1, 1] *= -1
        # surfel_render.set_transform(torch.from_numpy(inv_y))

        self.sensor_ui = SensorFrameUI("Sensor View")
        print("M - toggle play/step modes")
        print("N - steps one frame")

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            pass

    def _eval_accuracy(self):
        if self.gt_mesh is None:
            return

        rec_verts = self.fusion_ctx.get_stable_points().cpu()
        gt_closest, _ = self.oct_tree.query_closest_points(rec_verts)

        mesh_acc = mesh_accuracy(rec_verts, gt_closest)
        print("Current accuracy: {}".format(mesh_acc))

    def _run(self):
        frame_count = 0
        quit_flag = False
        read_next_frame = True
        use_camera_view = False

        while not quit_flag:
            if read_next_frame:
                print("Next frame: {}".format(frame_count))

                frame = self.sensor.next_frame()
                if frame is None:
                    continue

                frame_pcl = FramePointCloud(frame)

                device = "cuda:0"
                mask = frame_pcl.depth_mask.to(device)

                filtered_depth_image = bilateral_filter_depth_image(
                    torch.from_numpy(frame.depth_image).to(device),
                    mask, depth_scale=frame.info.depth_scale)
                
                frame_pcl.normals = estimate_normals(filtered_depth_image, frame.info,
                                                     mask).cpu()

                frame.normal_image = frame_pcl.normals
                self.sensor_ui.update(frame)

                self.rec_step.step(frame.info.kcam, frame_pcl)
                self.surfel_model.context.set_clear_color(0.32, 0.34, 0.87, 1)

                if use_camera_view:
                    self.viewer.set_camera_matrix(
                        frame.info.rt_cam.opengl_view_cam)

                read_next_frame = self.run_mode != RunMode.STEP
                frame_count += 1

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
                    self.surfel_render.set_max_confidence(20)
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

        cv2.destroyAllWindows()


