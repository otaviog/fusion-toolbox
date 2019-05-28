from queue import Empty
from multiprocessing import Lock
from enum import Enum
from cProfile import Profile

import torch
import torch.multiprocessing as mp
import numpy as np
from matplotlib.pyplot import get_cmap
import cv2

import tenviz

import fiontb.fusion.surfel
from fiontb.frame import FramePointCloud
from fiontb.filtering import blur_depth_image
from fiontb.viz.surfelrender import SurfelRender


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


def run_main_loop(sensor, rec_step, output_file, odometry=None, max_frames=None, single_process=False,
                  run_mode=RunMode.PLAY):
    torch.multiprocessing.set_start_method('spawn')

    surfels = fiontb.fusion.surfel.SurfelData(1024*1024*3, "cuda:0")
    surfels.share_memory()
    surfels_lock = Lock()

    if not single_process:
        surfel_update_queue = mp.Queue(5)
    else:
        surfel_update_queue = DummyQueue()

    frame_queue = mp.Queue()
    rec_loop = ReconstructionLoop(
        frame_queue,
        rec_step,
        init_args=(surfels, surfels_lock, surfel_update_queue, odometry))

    if not single_process:
        proc = mp.Process(target=rec_loop.run)
        import ipdb
        ipdb.set_trace()
        proc.start()
    else:
        rec_loop.init()

    context = tenviz.Context(640, 640)
    with context.current():
        surfel_render = SurfelRender(surfels)

    viewer = context.viewer(
        [surfel_render], cam_manip=tenviz.CameraManipulator.WASD)

    prof = Profile()
    prof.enable()

    inv_y = np.eye(4, dtype=np.float32)
    # inv_y[1, 1] *= -1
    surfel_render.set_transform(torch.from_numpy(inv_y))

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

                frame.depth_image = blur_depth_image(
                    frame.depth_image, 3, frame.depth_image > 0)
                frame_pcl = FramePointCloud(frame)

                sensor_ui.update(frame, frame_pcl.normals)
                live_pcl = frame_pcl.unordered_point_cloud(world_space=False)

                if not single_process:
                    frame_queue.put(
                        (frame.info.kcam, frame_pcl, live_pcl) if ret else None)
                else:
                    rec_loop.step((frame.info.kcam, frame_pcl, live_pcl))

                read_next_frame = run_mode != RunMode.STEP
            try:
                surfel_update, surfel_removal = surfel_update_queue.get_nowait()
                surfels_lock.acquire()
                if surfel_update.size(0) > 0:
                    # surfel_update_cpu = surfel_update
                    # surfel_update = surfel_update.to(surfels.device)
                    with context.current():
                        surfel_render.update(surfel_update)

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
                import ipdb
                ipdb.set_trace()

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
