"""Workflow for testing fusion using surfels.
"""

from queue import Empty
from multiprocessing import Lock
from enum import Enum

import numpy as np
import cv2
import open3d
import torch
import torch.multiprocessing as mp

import rflow
import tenviz

from fiontb.camera import RTCamera
from fiontb.frame import frame_to_pointcloud

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
            frame = self.frame_queue.get()
            if frame is None:
                break
            self.step_inst.step(frame)

    def step(self, frame):
        self.step_inst.step(frame)


class SurfelReconstructionStep(ReconstructionLoop):
    def __init__(self, surfels, surfels_lock, surfel_update_queue, odometry):
        self.surfels = surfels
        self.surfels_lock = surfels_lock
        self.surfel_update_queue = surfel_update_queue
        self.odometry = odometry
        self.count = 0

        self.fusion_ctx = fiontb.fusion.SurfelFusion(surfels)
        self.curr_rt_cam = RTCamera(np.eye(4, dtype=np.float32))

    def step(self, frame):
        live_pcl = frame_to_pointcloud(frame)

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
            live_pcl, self.curr_rt_cam)
        self.surfels_lock.release()
        self.surfel_update_queue.put(
            (surfel_update.cpu(), surfel_removal.cpu()))

        self.count += 1


class RunMode(Enum):
    PLAY = 0
    STEP = 1


def _main_loop(sensor, output_file, odometry=None, max_frames=None, single_process=False):
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
    viewer = context.viewer(cam_manip=tenviz.CameraManipulator.WASD)

    with context.current():
        render_surfels.points.from_tensor(surfels.points)
        render_surfels.normals.from_tensor(surfels.normals)
        render_surfels.colors.from_tensor(surfels.colors.byte())

    run_mode = RunMode.PLAY
    read_next_frame = True

    frame_count = 0
    try:
        while True:
            if frame_count == max_frames:
                break

            if read_next_frame:
                if frame_queue.empty():
                    frame, ret = sensor.next_frame()
                    # frame.depth_image = cv2.blur(frame.depth_image, (5, 5))
                    frame_queue.put(frame if ret else None)

                if single_process:
                    rec_loop.step(frame)

                read_next_frame = run_mode != RunMode.STEP
            try:
                surfel_update, surfel_removal = surfel_update_queue.get_nowait()
                surfel_update_cpu = surfel_update
                surfel_update = surfel_update.to(surfels.device)
                with context.current():
                    surfels_lock.acquire()
                    points = surfels.points[surfel_update]
                    render_surfels.update_bounds(points)
                    render_surfels.points[surfel_update] = points
                    render_surfels.normals[surfel_update] = surfels.normals[surfel_update]
                    colors = surfels.colors[surfel_update].byte()
                    render_surfels.colors[surfel_update] = colors
                    render_surfels.mark_visible(surfel_update_cpu)
                    # TODO: umark visible
                    surfels_lock.release()
            except Empty:
                pass

            viewer.draw(0)

            cv2.imshow("RGB Stream", cv2.cvtColor(
                frame.rgb_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("Depth Stream", frame.depth_image)

            key = cv2.waitKey(1)
            if key == 27:
                break
            key = chr(key & 0xff)

            if key == 'm':
                if run_mode == RunMode.PLAY:
                    run_mode = RunMode.STEP
                else:
                    run_mode = RunMode.PLAY
            elif key == 'n':
                read_next_frame = True
            frame_count += 1
    except KeyboardInterrupt:
        pass

    frame_queue.put(None)
    if not single_process:
        proc.join()

    cv2.destroyAllWindows()
    #dense_pcl = surfels.to_point_cloud()
    #tenviz.io.write_3dobject(output_file, dense_pcl.points,
    #normals=dense_pcl.normals, colors=dense_pcl.colors)


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
                   max_frames=None, single_process=False)


class SuperDenseFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        fusion = fiontb.fusion.surfel.DensePCLFusion(3, 0.2)

        _main_loop(sensor, fusion, resource.filepath, False)


@rflow.graph()
def test(g):
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


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()

    rflow.command.main()
