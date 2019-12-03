about:
	@echo "Empirical test tasks"

#########
# Frame
#

fiontb.camera-project:
	python3 -m fiontb._test.test_camera project


#########
# Viz
#

fiontb.viz-trajectoryviewer:
	python3 -m fiontb.viz.trajectoryviewer

#########
# Processing

fiontb.processing-bilateral:
	python3 -m fiontb._test.test_processing bilateral

fiontb.processing-normals:
	python3 -m fiontb._test.test_processing normals

fiontb.processing-erode_mask:
	python3 -m fiontb._test.test_processing erode_mask




#########
# Pose
#

fiontb.pose.open3d.rgbd-real:
	python3 -m fiontb.pose._test.test_open3dicp rgbd-real

fiontb.pose.open3d.rgbd-synthetic:
	python3 -m fiontb.pose._test.test_open3dicp rgbd-synthetic

fiontb.pose.open3d.rgb-real:
	python3 -m fiontb.pose._test.test_open3dicp rgb-real

fiontb.pose.open3d.rgb-synthetic:
	python3 -m fiontb.pose._test.test_open3dicp rgb-synthetic

fiontb.pose.open3d.coloricp-real:
	python3 -m fiontb.pose._test.test_open3dicp coloricp-real

fiontb.pose.open3d.coloricp-synthetic:
	python3 -m fiontb.pose._test.test_open3dicp coloricp-synthetic

fiontb.pose.open3d.rgbd-trajectory:
	python3 -m fiontb.pose._test.test_open3dicp rgbd-trajectory

fiontb.pose.icp.depth-real:
	python3 -m fiontb.pose._test.test_icp depth-real

fiontb.pose.icp.depth-synthetic:
	python3 -m fiontb.pose._test.test_icp depth-synthetic

fiontb.pose.icp.rgb-real:
	python3 -m fiontb.pose._test.test_icp rgb-real

fiontb.pose.icp.rgb-synthetic:
	python3 -m fiontb.pose._test.test_icp rgb-synthetic

fiontb.pose.icp.rgbd-real:
	python3 -m fiontb.pose._test.test_icp rgbd-real

fiontb.pose.icp.rgbd-synthetic:
	python3 -m fiontb.pose._test.test_icp rgbd-synthetic

fiontb.pose.icp.ms-depth-real:
	python3 -m fiontb.pose._test.test_icp ms-depth-real

fiontb.pose.icp.ms-depth-synthetic:
	python3 -m fiontb.pose._test.test_icp ms-depth-synthetic

fiontb.pose.icp.ms-rgb-real:
	python3 -m fiontb.pose._test.test_icp ms-rgb-real

fiontb.pose.icp.ms-rgb-synthetic:
	python3 -m fiontb.pose._test.test_icp ms-rgb-synthetic

fiontb.pose.icp.ms-rgbd-real:
	python3 -m fiontb.pose._test.test_icp ms-rgbd-real

fiontb.pose.icp.ms-rgbd-synthetic:
	python3 -m fiontb.pose._test.test_icp ms-rgbd-synthetic

fiontb.pose.icp.so3:
	python3 -m fiontb.pose._test.test_icp so3

fiontb.pose.icp.fail:
	python3 -m fiontb.pose._test.test_icp fail

fiontb.pose.icp.trajectory:
	python3 -m fiontb.pose._test.test_icp trajectory

fiontb.pose.autogradicp.depth-real:
	python3 -m fiontb.pose._test.test_autogradicp depth-real

fiontb.pose.autogradicp.depth-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp depth-synthetic

fiontb.pose.autogradicp.rgb-real:
	python3 -m fiontb.pose._test.test_autogradicp rgb-real

fiontb.pose.autogradicp.rgb-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp rgb-synthetic

fiontb.pose.autogradicp.rgbd-real:
	python3 -m fiontb.pose._test.test_autogradicp rgbd-real

fiontb.pose.autogradicp.rgbd-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp rgbd-synthetic

fiontb.pose.autogradicp.ms-depth-real:
	python3 -m fiontb.pose._test.test_autogradicp ms-depth-real

fiontb.pose.autogradicp.ms-depth-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp ms-depth-synthetic

fiontb.pose.autogradicp.ms-rgb-real:
	python3 -m fiontb.pose._test.test_autogradicp ms-rgb-real

fiontb.pose.autogradicp.ms-rgb-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp ms-rgb-synthetic

fiontb.pose.autogradicp.ms-rgbd-real:
	python3 -m fiontb.pose._test.test_autogradicp ms-rgbd-real

fiontb.pose.autogradicp.ms-rgbd-synthetic:
	python3 -m fiontb.pose._test.test_autogradicp ms-rgbd-synthetic

fiontb.pose.autogradicp.trajectory:
	python3 -m fiontb.pose._test.test_autogradicp trajectory

fiontb.pose.autogradicp.pcl-rgbd-real:
	python3 -m fiontb.pose._test.test_autogradicp pcl-rgbd-real

#########
# Surfel
#

fiontb.surfel-adding:
	python3 -m fiontb._test.test_surfel adding

fiontb.surfel-transform:
	python3 -m fiontb._test.test_surfel transform

fiontb.surfel-downsample:
	python3 -m fiontb._test.test_surfel downsample

fiontb.surfel-merge:
	python3 -m fiontb._test.test_surfel merge

fiontb.viz.surfelrender:
	python3 -m fiontb.viz.surfelrender

fiontb.fusion.surfel.indexmap-surfel-raster:
	python3 -m fiontb.fusion.surfel._test.test_indexmap surfel-raster

fiontb.fusion.surfel.update-vanilla:
	python3 -m fiontb.fusion.surfel._test.test_update vanilla

fiontb.fusion.surfel.update-ef_like:
	python3 -m fiontb.fusion.surfel._test.test_update ef_like

fiontb.fusion.surfel.merge:
	python3 -m fiontb.fusion.surfel._test.test_merge

fiontb.fusion.surfel.carve_space:
	python3 -m fiontb.fusion.surfel._test.test_carve_space

fiontb.fusion.surfel.remove_unstable:
	python3 -m fiontb.fusion.surfel._test.test_remove_unstable

fiontb.fusion.surfel.clean:
	python3 -m fiontb.fusion.surfel._test.test_clean

fiontb.fusion.surfel.fusion:
	python3 -m fiontb.fusion.surfel._test.test_fusion

fiontb.fusion.surfel.effusion:
	python3 -m fiontb.fusion.surfel._test.test_effusion

fiontb.fusion.surfel.model-tracking:
	python3 -m fiontb.fusion.surfel._test.test_model_tracking

fiontb.pipeline.surfel_slam.real-scene:
	python3 -m fiontb.pipeline._test.test_surfel_slam real_scene

fiontb.pipeline.surfel_slam.synthetic-scene:
	python3 -m fiontb.pipeline._test.test_surfel_slam synthetic_scene

# FSF

fiontb.fusion.fsf.registration:	
	python3 -m fiontb.fusion.fsf._test.test_registration

fiontb.fusion.fsf.fusion:	
	python3 -m fiontb.fusion.fsf._test.test_fusion

fiontb.fusion.fsf.merge:
	python3 -m fiontb.fusion.fsf._test.test_merge

fiontb.pipeline.fsf-real-scene:
	python3 -m fiontb.pipeline._test.test_fsf_slam real_scene

fiontb.pipeline.fsf-synthetic-scene:
	python3 -m fiontb.pipeline._test.test_fsf_slam synthetic_scene
