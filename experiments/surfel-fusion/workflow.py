"""Workflow for testing fusion using surfels.
"""

import cProfile
from queue import Empty
from multiprocessing import Lock

import rflow
import numpy as np
# import cv2

#import open3d
import torch
import torch.multiprocessing as mp

from fiontb import PointCloud
from fiontb.camera import RTCamera
from fiontb.frame import FramePoints, compute_normals

import fiontb.sensor
import fiontb.fusion
import fiontb.fusion.surfel
# import fiontb.pose
# import fiontb.pose.icp

import tenviz


def fusion_loop(queue, lock, sensor, surfels, odometry=None, output_file=None, max_frames=None):
    fusion_ctx = fiontb.fusion.SurfelFusion(surfels)
    curr_rt_cam = RTCamera(np.eye(4))
    count = 0
    print("Has cuda on subprocess:", torch.cuda.is_available())
    # torch.from_numpy(points.camera_points).to("cuda:0")

    while True:
        frame, ret = sensor.next_frame()
        # frame.depth_image = cv2.blur(frame.depth_image, (5, 5))

        if not ret:
            break

        points = FramePoints(frame)
        normals = compute_normals(points.depth_image)
        normals = normals.reshape(-1, 3)
        normals = normals[points.fg_mask.flatten()]

        live_pcl = PointCloud(points.camera_points,
                              points.colors, normals)

        if odometry is not None:
            curr_rt_cam = RTCamera(
                np.array(odometry[count]['rt_cam'], dtype=np.float32))
        else:
            fusion_pcl = fusion_ctx.get_odometry_model()
            if not fusion_pcl.is_empty():
                relative_rt_cam = fiontb.pose.icp.estimate_icp_geo(
                    live_pcl, fusion_pcl)
                curr_rt_cam.integrate(relative_rt_cam)

        live_pcl.transform(curr_rt_cam.cam_to_world)

        lock.acquire()
        surfel_update, surfel_removal = fusion_ctx.fuse(live_pcl, curr_rt_cam)
        lock.release()
        queue.put((surfel_update, surfel_removal))
        count += 1
        if max_frames is not None and count >= max_frames:
            break
    dense_pcl = surfels.to_point_cloud()
    tenviz.io.write_3dobject(output_file, dense_pcl.points,
                               normals=dense_pcl.normals, colors=dense_pcl.colors)


def _main_loop0(sensor, output_file, show, odometry=None, max_frames=None):
    torch.multiprocessing.set_start_method('spawn')
    surfels = fiontb.fusion.surfel.SceneSurfelData(1024*1024*3, "cuda:0")

    queue = mp.Queue()
    lock = Lock()
    
    surfels.share_memory()

    proc = mp.Process(target=fusion_loop, args=(
        queue, lock, sensor, surfels, odometry, output_file, max_frames))

    #fusion_loop(queue, sensor, surfels, odometry, output_file, max_frames)

    #queue.put((torch.arange(0, 124, dtype=torch.int64),
    #torch.tensor([], dtype=torch.int64)))

    context = tenviz.Context(640, 640)
    render_surfels = context.add_surfel_cloud()
    viewer = context.viewer()

    with context.current():
        render_surfels.points.from_tensor(surfels.points)
        render_surfels.normals.from_tensor(surfels.normals)
        render_surfels.colors.from_tensor(surfels.colors.byte())

    import ipdb; ipdb.set_trace()
    proc.start()
    import time
    while True:
        # continue
        # frame, ret = sensor.next_frame()
        # frame.depth_image = cv2.blur(frame.depth_image, (5, 5))

        try:
            surfel_update, surfel_removal = queue.get_nowait()
            with context.current():
                
                lock.acquire()
                t0 = time.time()
                render_surfels.points[surfel_update] = surfels.points[surfel_update]
                render_surfels.normals[surfel_update] = surfels.normals[surfel_update]
                render_surfels.colors[surfel_update] = surfels.colors[surfel_update].byte()
                print(time.time() - t0)
                lock.release()

                render_surfels.mark_visible(surfel_update)

            # render_surfels.unmarkvisibl
        except Empty:
            pass

        viewer.draw(0)

        if False:
            cv2.imshow("RGB Stream", cv2.cvtColor(
                frame.rgb_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("Depth Stream", frame.depth_image)

            key = cv2.waitKey(1)
            key = chr(key & 0xff)
            if key == 'n':
                break


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

            points = FramePoints(frame)

            live_pcl = PointCloud(points.camera_points, points.colors)
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
        _main_loop0(sensor, resource.filepath, True, odometry, max_frames=None)


class SuperDenseFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        fusion = fiontb.fusion.surfel.DensePCLFusion(3, 0.2)

        _main_loop(sensor, fusion, resource.filepath, True)


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
    import ipdb; ipdb.set_trace()

    rflow.command.main()
