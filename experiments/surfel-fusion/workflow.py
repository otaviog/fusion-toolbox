"""Workflow for testing fusion using surfels.
"""

from queue import Empty
from multiprocessing import Lock
from enum import Enum
from cProfile import Profile


import numpy as np
import cv2
import open3d
import torch
import torch.multiprocessing as mp
from matplotlib.pyplot import get_cmap

import rflow
import tenviz

from fiontb.datatypes import PointCloud
from fiontb.camera import RTCamera
from fiontb.frame import frame_to_pointcloud, compute_normals, FramePoints
import fiontb.fiontblib as fiontblib

import fiontb.sensor
import fiontb.fusion
import fiontb.fusion.surfel
import fiontb.pose
import fiontb.pose.icp


class DummyQueue:
    def __init__(self):
        self.queue = []

    def put(self, value):
        self.queue.append(value)

    def get_nowait(self):
        if not self.queue:
            raise Empty()

        value = self.queue[0]
        self.queue = self.queue[1:]

        return value

    def get(self):
        return self.get_nowait()

    def full(self):
        return False


class ReconstructionLoop:
    def __init__(self, frame_queue, step_class, init_args=()):
        self.step_class = step_class
        self.step_class_args = init_args
        self.frame_queue = frame_queue
        self.step_inst = None

    def init(self):
        self.step_inst = self.step_class(*self.step_class_args)

    def run(self):
        self.init()
        while True:
            frame_data = self.frame_queue.get()
            if frame_data is None:
                break
            self.step_inst.step(*frame_data)

    def step(self, frame_data):
        self.step_inst.step(*frame_data)


class SurfelReconstructionStep(ReconstructionLoop):
    def __init__(self, surfels, surfels_lock, surfel_update_queue, odometry):
        self.surfels = surfels
        self.surfels_lock = surfels_lock
        self.surfel_update_queue = surfel_update_queue
        self.odometry = odometry
        self.count = 0

        self.fusion_ctx = fiontb.fusion.SurfelFusion(surfels)
        self.curr_rt_cam = RTCamera(np.eye(4, dtype=np.float32))

    def step(self, kcam, frame_points, live_pcl):
        if self.odometry is not None:
            self.curr_rt_cam = RTCamera(
                np.array(self.odometry[self.count]['rt_cam'], dtype=np.float32))
        else:
            fusion_pcl = self.fusion_ctx.get_odometry_model()
            if not fusion_pcl.is_empty():
                relative_rt_cam = fiontb.pose.icp.estimate_icp_geo(
                    live_pcl, fusion_pcl)
                self.curr_rt_cam.integrate(relative_rt_cam)

        live_pcl.transform(self.curr_rt_cam.cam_to_world)

        self.surfels_lock.acquire()
        surfel_update, surfel_removal = self.fusion_ctx.fuse(
            frame_points, live_pcl, kcam, self.curr_rt_cam)
        self.surfels_lock.release()
        self.surfel_update_queue.put(
            (surfel_update.cpu(), surfel_removal.cpu()))

        self.count += 1


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

        cmap = get_cmap('viridis', self.frame.info.depth_max)
        depth_img = (self.frame.depth_image / self.frame.info.depth_max)
        depth_img = cmap(depth_img)
        depth_img = depth_img[:, :, 0:3]
        depth_img = (depth_img*255).astype(np.uint8)

        depth_alpha = cv2.getTrackbarPos(
            SensorFrameUI._DEPTH_OPPACITY_LABEL, self.title) / 100.0

        canvas = cv2.addWeighted(depth_img, depth_alpha,
                                 self.frame.rgb_image, 1.0 - depth_alpha, 0.0)

        normal_alpha = cv2.getTrackbarPos(
            SensorFrameUI._NORMAL_OPPACITY_LABEL, self.title) / 100.0

        canvas = cv2.addWeighted(self.normal_image, normal_alpha,
                                 canvas, 1.0 - normal_alpha, 0.0)
        cv2.imshow(self.title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def update(self, frame, normals):
        self.frame = frame
        self.normal_image = (normals + 1)*0.5*255
        self.normal_image = self.normal_image.astype(np.uint8)
        self._update(0)


def _main_loop(sensor, output_file, odometry=None, max_frames=None, single_process=False,
               run_mode=RunMode.PLAY):
    torch.multiprocessing.set_start_method('spawn')
    
    surfels = fiontb.fusion.surfel.SceneSurfelData(1024*1024*3, "cuda:0")
    surfels.share_memory()
    surfels_lock = Lock()

    if not single_process:
        surfel_update_queue = mp.Queue(5)
    else:
        surfel_update_queue = DummyQueue()

    frame_queue = mp.Queue()
    rec_loop = ReconstructionLoop(
        frame_queue,
        SurfelReconstructionStep,
        init_args=(surfels, surfels_lock, surfel_update_queue, odometry))

    if not single_process:
        proc = mp.Process(target=rec_loop.run)
        import ipdb
        ipdb.set_trace()
        proc.start()
    else:
        rec_loop.init()

    context = tenviz.Context(640, 640)
    render_surfels = context.add_surfel_cloud()
    viewer = context.viewer(
        [render_surfels], cam_manip=tenviz.CameraManipulator.WASD)

    prof = Profile()
    prof.enable()

    with context.current():
        render_surfels.points.from_tensor(surfels.points)
        render_surfels.normals.from_tensor(surfels.normals)
        render_surfels.colors.from_tensor(surfels.colors.byte())
        render_surfels.radii.from_tensor(surfels.radii.view(-1, 1))
        render_surfels.counts.from_tensor(surfels.counts.view(-1, 1))

    inv_y = np.eye(4, dtype=np.float32)
    inv_y[1, 1] *= -1
    render_surfels.set_transform(torch.from_numpy(inv_y))

    read_next_frame = True

    sensor_ui = SensorFrameUI("Sensor View")
    print("M - toggle play/step modes")
    print("N - steps one frame")
    frame_count = 0
    try:
        while True:
            if frame_count == max_frames:
                break

            if read_next_frame and frame_queue.empty():
                print("Next frame")
                frame, ret = sensor.next_frame()
                no_blur_depth_image = frame.depth_image
                frame.depth_image = cv2.blur(frame.depth_image, (3, 3))

                points = FramePoints(frame)
                normals = fiontblib.calculate_depth_normals(
                    torch.from_numpy(points.camera_xyz_image),
                    torch.from_numpy(points.fg_mask.astype(np.uint8))).numpy()

                sensor_ui.update(frame, normals)

                frame.depth_image = no_blur_depth_image
                points = FramePoints(frame)
                live_pcl = PointCloud(points.camera_points, points.colors,
                                      normals.reshape(-1, 3)[points.fg_mask.flatten()])

                if not single_process:
                    frame_queue.put(
                        (frame.info.kcam, points, live_pcl) if ret else None)
                else:
                    rec_loop.step((frame.info.kcam, points, live_pcl))

                read_next_frame = run_mode != RunMode.STEP
            try:
                surfel_update, surfel_removal = surfel_update_queue.get_nowait()
                surfels_lock.acquire()
                if surfel_update.size(0) > 0:
                    surfel_update_cpu = surfel_update
                    surfel_update = surfel_update.to(surfels.device)
                    with context.current():
                        points = surfels.points[surfel_update]

                        render_surfels.update_bounds(points)
                        render_surfels.points[surfel_update] = points
                        render_surfels.normals[surfel_update] = surfels.normals[surfel_update]
                        colors = surfels.colors[surfel_update].byte()
                        render_surfels.colors[surfel_update] = colors
                        render_surfels.radii[surfel_update] = surfels.radii[surfel_update].view(
                            -1, 1)
                        render_surfels.counts[surfel_update] = surfels.counts[surfel_update].view(
                            -1, 1)
                        render_surfels.mark_visible(surfel_update_cpu)

                render_surfels.mark_invisible(surfel_removal.cpu())
                surfels_lock.release()
            except Empty:
                pass

            def _handle_key(key):
                nonlocal run_mode
                nonlocal read_next_frame

                if key == 27:
                    return False
                key = chr(key & 0xff).lower()

                if key == 'm':
                    if run_mode == RunMode.PLAY:
                        run_mode = RunMode.STEP
                    else:
                        run_mode = RunMode.PLAY
                elif key == 'n':
                    read_next_frame = True
                
                return True

            key = viewer.draw(0)
            if not _handle_key(key):
                break

            key = cv2.waitKey(1)
            if not _handle_key(key):
                break

            if chr(key & 0xff) == 't':
                import ipdb; ipdb.set_trace()

            frame_count += 1
    except KeyboardInterrupt:
        pass

    frame_queue.put(None)
    if not single_process:
        proc.join()

    cv2.destroyAllWindows()
    prof.disable()
    prof.dump_stats("profile.prof")
    # dense_pcl = surfels.to_point_cloud()
    # tenviz.io.write_3dobject(output_file, dense_pcl.points,
    # normals=dense_pcl.normals, colors=dense_pcl.colors)


class FrameToFrameOdometry(rflow.Interface):
    def evaluate(self, resource, dataset):
        import copy
        import json

        sensor = fiontb.sensor.DatasetSensor(dataset)

        accum_rt_cam = RTCamera(np.eye(4))
        last_pcl = None

        trajectory = []

        c = 0
        while True:
            frame, ret = sensor.next_frame()
            if not ret:
                break

            live_pcl = frame_to_pointcloud(frame)
            # live_pcl.transform(accum_rt_cam.cam_to_world)

            if last_pcl is not None:
                relative_rt_cam = fiontb.pose.icp.estimate_icp_geo(
                    live_pcl, last_pcl)
                accum_rt_cam.integrate(relative_rt_cam)

            last_pcl = live_pcl
            trajectory.append(copy.deepcopy(accum_rt_cam))
            c += 1

        with open(resource.filepath, 'w') as file:
            trajectory = [{'rt_cam': rt_cam.matrix.tolist()}
                          for rt_cam in trajectory]

            json.dump(trajectory, file, indent=1)

        return trajectory

    def load(self, resource):
        import json

        with open(resource.filepath, 'r') as file:
            trajectory = json.load(file)

        return trajectory


class ViewOdometry(rflow.Interface):
    def evaluate(self, dataset, odometry):
        from fiontb.viz.datasetviewer import DatasetViewer

        for frame_info, rt_cam in zip(dataset.frame_infos, odometry):
            frame_info['rt_cam'] = rt_cam['rt_cam']

        viewer = DatasetViewer(dataset, title="Frame to frame odometry")
        viewer.run()


class SurfelFusion(rflow.Interface):
    def evaluate(self, resource, dataset, odometry):
        sensor = fiontb.sensor.DatasetSensor(dataset)
        _main_loop(sensor, resource.filepath, odometry,
                   max_frames=None, single_process=True, run_mode=RunMode.STEP)


class SuperDenseFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        fusion = fiontb.fusion.surfel.DensePCLFusion(3, 0.2)

        _main_loop(sensor, fusion, resource.filepath, False)


@rflow.graph()
def scene3(g):
    ds_g = rflow.open_graph("../../test-data/rgbd/scene3", "sample")

    g.frame_to_frame = FrameToFrameOdometry(
        rflow.FSResource("frame2frame.json"))
    with g.frame_to_frame as args:
        args.dataset = ds_g.to_ftb

    g.view_frame_to_frame = ViewOdometry()
    with g.view_frame_to_frame as args:
        args.dataset = ds_g.to_ftb
        args.odometry = g.frame_to_frame

    g.fusion = SurfelFusion(rflow.FSResource("scene3.ply"))
    with g.fusion as args:
        args.dataset = ds_g.to_ftb
        args.odometry = g.frame_to_frame

    g.dense_fusion = SuperDenseFusion(rflow.FSResource("scene3.ply"))
    with g.dense_fusion as args:
        args.dataset = ds_g.to_ftb


@rflow.graph()
def iclnuim(g):
    ds_g = rflow.open_graph("../../test-data/rgbd/iclnuim", "iclnuim")

    g.frame_to_frame = FrameToFrameOdometry(
        rflow.FSResource("lr0-frame2frame.json"))
    with g.frame_to_frame as args:
        args.dataset = ds_g.lr0_to_ftb

    g.view_frame_to_frame = ViewOdometry()
    with g.view_frame_to_frame as args:
        args.dataset = ds_g.lr0_to_ftb
        args.odometry = g.frame_to_frame

    g.fusion = SurfelFusion(rflow.FSResource("lr0.ply"))
    with g.fusion as args:
        args.dataset = ds_g.lr0_to_ftb
        args.odometry = g.frame_to_frame

    g.dense_fusion = SuperDenseFusion(rflow.FSResource("lr0.ply"))
    with g.dense_fusion as args:
        args.dataset = ds_g.lr0_to_ftb


if __name__ == '__main__':
    rflow.command.main()
