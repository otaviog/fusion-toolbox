
import numpy as np

import rflow

import fiontb.sensor
from fiontb.frame import FramePointCloud
from fiontb.camera import RTCamera


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

            live_pcl = FramePointCloud(frame).to_point_cloud(world_space=False)
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
