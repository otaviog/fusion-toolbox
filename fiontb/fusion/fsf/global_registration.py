import torch

import open3d


class GlobalRegistration:
    def __init__(self, voxel_size=5e-3):
        self.voxel_size = voxel_size
        self.radius = voxel_size*5

    def estimate(self, target_surfels, local_surfels):
        target_pcl = target_surfels.to_open3d()
        local_pcl = local_surfels.to_open3d()

        target_pcl = open3d.voxel_down_sample(target_pcl, self.voxel_size)
        local_pcl = open3d.voxel_down_sample(local_pcl, self.voxel_size)

        print(target_pcl, local_pcl)
        kdtree_param = open3d.geometry.KDTreeSearchParamHybrid(
            radius=self.radius, max_nn=100)
        target_features = open3d.registration.compute_fpfh_feature(
            target_pcl, kdtree_param)
        local_features = open3d.registration.compute_fpfh_feature(
            local_pcl, kdtree_param)
        dist_thresh = self.voxel_size * 5.5
        if True:
            result = open3d.registration.registration_ransac_based_on_feature_matching(
                local_pcl, target_pcl, local_features, target_features,
                dist_thresh,  open3d.registration.TransformationEstimationPointToPoint(False), 4, [
                    open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9),
                    open3d.registration.CorrespondenceCheckerBasedOnDistance(
                        dist_thresh)
                ], open3d.registration.RANSACConvergenceCriteria(4000000, 500))

            #open3d.registration.regi
        print(result.transformation)
        return torch.from_numpy(result.transformation).float()
