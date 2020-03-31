about:
	@echo "Empirical test tasks"

#########
# Frame
#

slamtb.camera-project:
	python3 -m slamtb._test.test_camera project


#########
# Viz
#

slamtb.viz-trajectoryviewer:
	python3 -m slamtb.viz.trajectoryviewer

#########
# Processing

slamtb.processing-bilateral:
	python3 -m slamtb._test.test_processing bilateral

slamtb.processing-normals:
	python3 -m slamtb._test.test_processing normals

slamtb.processing-erode_mask:
	python3 -m slamtb._test.test_processing erode_mask


#########
# Registration
#

slamtb.registration.icp.depth-real:
	python3 -m slamtb.registration._test.test_icp depth-real

slamtb.registration.icp.depth-synthetic:
	python3 -m slamtb.registration._test.test_icp depth-synthetic

slamtb.registration.icp.rgb-real:
	python3 -m slamtb.registration._test.test_icp rgb-real

slamtb.registration.icp.rgb-synthetic:
	python3 -m slamtb.registration._test.test_icp rgb-synthetic

slamtb.registration.icp.rgbd-real:
	python3 -m slamtb.registration._test.test_icp rgbd-real

slamtb.registration.icp.rgbd-synthetic:
	python3 -m slamtb.registration._test.test_icp rgbd-synthetic

slamtb.registration.icp.ms-depth-real:
	python3 -m slamtb.registration._test.test_icp ms-depth-real

slamtb.registration.icp.ms-depth-synthetic:
	python3 -m slamtb.registration._test.test_icp ms-depth-synthetic

slamtb.registration.icp.ms-rgb-real:
	python3 -m slamtb.registration._test.test_icp ms-rgb-real

slamtb.registration.icp.ms-rgb-synthetic:
	python3 -m slamtb.registration._test.test_icp ms-rgb-synthetic

slamtb.registration.icp.ms-rgbd-real:
	python3 -m slamtb.registration._test.test_icp ms-rgbd-real

slamtb.registration.icp.ms-rgbd-synthetic:
	python3 -m slamtb.registration._test.test_icp ms-rgbd-synthetic

slamtb.registration.icp.so3:
	python3 -m slamtb.registration._test.test_icp so3

slamtb.registration.icp.fail:
	python3 -m slamtb.registration._test.test_icp fail

slamtb.registration.icp.trajectory:
	python3 -m slamtb.registration._test.test_icp trajectory

slamtb.registration.autogradicp.depth-real:
	python3 -m slamtb.registration._test.test_autogradicp depth-real

slamtb.registration.autogradicp.depth-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp depth-synthetic

slamtb.registration.autogradicp.rgb-real:
	python3 -m slamtb.registration._test.test_autogradicp rgb-real

slamtb.registration.autogradicp.rgb-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp rgb-synthetic

slamtb.registration.autogradicp.rgbd-real:
	python3 -m slamtb.registration._test.test_autogradicp rgbd-real

slamtb.registration.autogradicp.rgbd-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp rgbd-synthetic

slamtb.registration.autogradicp.ms-depth-real:
	python3 -m slamtb.registration._test.test_autogradicp ms-depth-real

slamtb.registration.autogradicp.ms-depth-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp ms-depth-synthetic

slamtb.registration.autogradicp.ms-rgb-real:
	python3 -m slamtb.registration._test.test_autogradicp ms-rgb-real

slamtb.registration.autogradicp.ms-rgb-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp ms-rgb-synthetic

slamtb.registration.autogradicp.ms-rgbd-real:
	python3 -m slamtb.registration._test.test_autogradicp ms-rgbd-real

slamtb.registration.autogradicp.ms-rgbd-synthetic:
	python3 -m slamtb.registration._test.test_autogradicp ms-rgbd-synthetic

slamtb.registration.autogradicp.trajectory:
	python3 -m slamtb.registration._test.test_autogradicp trajectory

slamtb.registration.autogradicp.pcl-rgb-real:
	python3 -m slamtb.registration._test.test_autogradicp pcl-rgb-real

slamtb.registration.autogradicp.pcl-ms-rgbd-real:
	python3 -m slamtb.registration._test.test_autogradicp pcl-ms-rgbd-real

#########
# Surfel
#

slamtb.surfel-adding:
	python3 -m slamtb._test.test_surfel adding

slamtb.surfel-transform:
	python3 -m slamtb._test.test_surfel transform

slamtb.surfel-downsample:
	python3 -m slamtb._test.test_surfel downsample

slamtb.surfel-merge:
	python3 -m slamtb._test.test_surfel merge

slamtb.viz.surfelrender:
	python3 -m slamtb.viz.surfelrender

slamtb.fusion.surfel.indexmap-surfel-raster:
	python3 -m slamtb.fusion.surfel._test.test_indexmap surfel-raster

slamtb.fusion.surfel.update-vanilla:
	python3 -m slamtb.fusion.surfel._test.test_update vanilla

slamtb.fusion.surfel.update-ef_like:
	python3 -m slamtb.fusion.surfel._test.test_update ef_like

slamtb.fusion.surfel.merge:
	python3 -m slamtb.fusion.surfel._test.test_merge

slamtb.fusion.surfel.carve_space:
	python3 -m slamtb.fusion.surfel._test.test_carve_space

slamtb.fusion.surfel.clean:
	python3 -m slamtb.fusion.surfel._test.test_clean

slamtb.fusion.surfel.fusion:
	python3 -m slamtb.fusion.surfel._test.test_fusion

slamtb.fusion.surfel.effusion:
	python3 -m slamtb.fusion.surfel._test.test_effusion

slamtb.pipeline.surfel_slam.real-scene:
	python3 -m slamtb.pipeline._test.test_surfel_slam real_scene

slamtb.pipeline.surfel_slam.synthetic-scene:
	python3 -m slamtb.pipeline._test.test_surfel_slam synthetic_scene
