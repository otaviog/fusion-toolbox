from queue import Empty
from multiprocessing import Lock
from enum import Enum

import torch
import torch.multiprocessing as mp
import numpy as np
from matplotlib.pyplot import get_cmap
import cv2

import tenviz

import fiontb.fusion.surfel
from fiontb.frame import FramePointCloud
from fiontb.filtering import blur_depth_image
from fiontb.viz.surfelrender import SurfelRender, RenderMode


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


class MainLoop:
    def __init__(self, sensor, rec_step, output_file,
                 odometry=None, max_frames=None, single_process=False,
                 run_mode=RunMode.PLAY):
        torch.multiprocessing.set_start_method('spawn')
        self.sensor = sensor
        self.surfels = fiontb.fusion.surfel.SurfelData(1024*1024*3, "cuda:0")
        # surfels.share_memory()
        self.surfels_lock = Lock()
        self.max_frames = max_frames
        self.is_single_process = single_process
        self.run_mode = run_mode

        if not single_process:
            self.surfel_update_queue = mp.Queue(5)
        else:
            self.surfel_update_queue = DummyQueue()

        self.frame_queue = mp.Queue()
        self.rec_loop = ReconstructionLoop(
            self.frame_queue,
            rec_step,
            init_args=(self.surfels, self.surfels_lock, self.surfel_update_queue, odometry))

        if not single_process:
            proc = mp.Process(target=self.rec_loop.run)
            import ipdb
            ipdb.set_trace()
            proc.start()
        else:
            self.rec_loop.init()

        self.context = tenviz.Context(640, 640)
        with self.context.current():
            self.surfel_render = SurfelRender(self.surfels)

        self.viewer = self.context.viewer(
            [self.surfel_render], cam_manip=tenviz.CameraManipulator.WASD)
        self.viewer.reset_view()

        inv_y = np.eye(4, dtype=np.float32)
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

        self.frame_queue.put(None)
        if not self.is_single_process:
            self.proc.join()

    def _run(self):
        frame_count = 0
        quit_flag = False
        read_next_frame = True
        while not quit_flag:
            if frame_count == self.max_frames:
                break

            if read_next_frame and self.frame_queue.empty():
                print("Next frame: {}".format(frame_count))

                frame, ret = self.sensor.next_frame()

                if frame is None:
                    continue

                frame.depth_image = blur_depth_image(
                    frame.depth_image, 7, frame.depth_image > 0)
                frame_pcl = FramePointCloud(frame)

                self.sensor_ui.update(frame, frame_pcl.normals)
                live_pcl = frame_pcl.unordered_point_cloud(world_space=False)

                if not self.is_single_process:
                    self.frame_queue.put(
                        (frame.info.kcam, frame_pcl, live_pcl) if ret else None)
                else:
                    self.rec_loop.step((frame.info.kcam, frame_pcl, live_pcl))

                read_next_frame = self.run_mode != RunMode.STEP
                frame_count += 1

            try:
                surfel_update, surfel_removal = self.surfel_update_queue.get_nowait()
                print("{} surfels updated, {} removed".format(surfel_update.numel(),
                                                              surfel_removal.numel()))
                self.surfels_lock.acquire()

                with self.context.current():
                    self.surfel_render.update(surfel_update)

                self.surfels_lock.release()
            except Empty:
                pass

            keys = [self.viewer.wait_key(0), cv2.waitKey(1)]
            for key in keys:
                key = key & 0xff
                if key == 27:
                    quit_flag = True

                key = chr(key).lower()

                if key == 'm':
                    if run_mode == RunMode.PLAY:
                        run_mode = RunMode.STEP
                    else:
                        run_mode = RunMode.PLAY
                elif key == 'q':
                    quit_flag = True
                elif key == 'n':
                    read_next_frame = True
                elif key == 'i':
                    with self.context.current():
                        self.surfel_render.set_render_mode(RenderMode.Confs)
                elif key == 'u':
                    with self.context.current():
                        self.surfel_render.set_render_mode(RenderMode.Color)
                elif key == 'o':
                    with self.context.current():
                        self.surfel_render.set_render_mode(RenderMode.Normal)
                elif key == 'p':
                    with self.context.current():
                        self.surfel_render.set_stable_threshold(10)
                elif key == 'l':
                    with self.context.current():
                        self.surfel_render.set_stable_threshold(-1)
                elif key == 'b':
                    import ipdb
                    ipdb.set_trace()

            if keys[0] < 0:
                quit_flag = True

        cv2.destroyAllWindows()
