import rflow
import numpy as np
import cv2

from fiontb import PointCloud
from fiontb.camera import RTCamera
import fiontb.data.sensor
import fiontb.fusion
import fiontb.fusion.surfel
import fiontb.pose
import fiontb.pose.icp

import shapelab.io
import cProfile


def _main_loop(sensor, fusion_ctx, output_file, show=False):
    prof = cProfile.Profile()
    prof.enable()

    view = fiontb.fusion.surfel.SurfelView()
    last_rt_cam = RTCamera(np.eye(4))
    c = 0
    while True:
        snap, ret = sensor.next_frame()
        if not ret:
            break

        live_pcl = PointCloud(snap.get_cam_points(), snap.colors)
        live_pcl.transform(last_rt_cam.cam_to_world)

        fusion_pcl = fusion_ctx.get_odometry_model()
        if not fusion_pcl.is_empty():
            last_rt_cam = fiontb.pose.icp.estimate_icp_geo(
                live_pcl, fusion_pcl)
        else:
            live_pcl.to_open3d(compute_normals=True)

        live_pcl.transform(last_rt_cam.cam_to_world)

        fusion_ctx.fuse(live_pcl)
        c += 1
        print(c)

        if show:
            view.update(fusion_ctx.get_model())

        while show:
            view.viewer.draw(0)
            cv2.imshow("RGB Stream", cv2.cvtColor(
                snap.rgb_image, cv2.COLOR_RGB2BGR))
            cv2.imshow("Depth Stream", snap.depth_image)

            key = cv2.waitKey(5)
            key = chr(key & 0xff)
            if key == 'n':
                break

    prof.disable()
    prof.dump_stats("view.prof")
    dense_pcl = fusion_ctx.get_model()
    shapelab.io.write_3dobject(output_file, dense_pcl.points,
                               normals=dense_pcl.normals, colors=dense_pcl.colors)


class SurfelFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        fusion_ctx = fiontb.fusion.SurfelFusion()
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        _main_loop(sensor, fusion_ctx, resource.filepath, True)


class SuperDenseFusion(rflow.Interface):
    def evaluate(self, resource, dataset):
        sensor = fiontb.data.sensor.DatasetSensor(dataset)
        fusion = fiontb.fusion.surfel.DensePCLFusion(3, 0.2)

        _main_loop(sensor, fusion, resource.filepath, True)


@rflow.graph()
def test(g):
    ds_g = rflow.open_graph("../../test-data/rgbd/scene3", "sample")

    g.fusion = SurfelFusion(rflow.FSResource("scene3.ply"))
    with g.fusion as args:
        args.dataset = ds_g.dataset

    g.dense_fusion = SuperDenseFusion(rflow.FSResource("scene3.ply"))
    with g.dense_fusion as args:
        args.dataset = ds_g.dataset


if __name__ == '__main__':
    rflow.command.main()
