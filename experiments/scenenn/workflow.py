import rflow
import open3d
import numpy as np

import fusionkit
from fusionkit.data.scenenn import load_scenenn
from fusionkit.viz.datasetviewer import DatasetViewer


class ViewSceneNN(rflow.Interface):
    def evaluate(self):
        innet_traj = load_scenenn("030/030.oni", "030/trajectory.log", "asus")

        viewer = DatasetViewer(innet_traj)
        viewer.run()


@rflow.graph()
def view(g):
    g.main = ViewSceneNN()


def draw_registration_result(source, target, transformation):
    import copy

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = open3d.voxel_down_sample(pcd, voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.compute_fpfh_feature(
        pcd_down, open3d.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(
        source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = open3d.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        distance_threshold,
        open3d.TransformationEstimationPointToPoint(False), 4,
        [open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.RANSACConvergenceCriteria(4000000, 500))
    return result


class RegisterPCL(rflow.Interface):
    def evaluate(self):
        scenenn = load_scenenn("030/030.oni", "030/trajectory.log", "asus")

        snap0 = scenenn[0]
        snap1 = scenenn[1]

        p0 = fusionkit.to_open3d(snap0.cam_points, snap0.colors)
        p1 = fusionkit.to_open3d(snap1.cam_points, snap1.colors)

        p0_d, p0_fp = preprocess_point_cloud(p0, 0.05)
        p1_d, p1_fp = preprocess_point_cloud(p1, 0.05)

        reg = execute_global_registration(p0_d, p1_d, p0_fp, p1_fp, 0.05)

        draw_registration_result(p0, p1, reg.transformation)
        print(reg.transformation)

@rflow.graph()
def reg(g):
    g.main = RegisterPCL()


if __name__ == '__main__':
    rflow.command.main()
