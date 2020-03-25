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
# Registration
#

fiontb.registration.icp.depth-real:
	python3 -m fiontb.registration._test.test_icp depth-real

fiontb.registration.icp.depth-synthetic:
	python3 -m fiontb.registration._test.test_icp depth-synthetic

fiontb.registration.icp.rgb-real:
	python3 -m fiontb.registration._test.test_icp rgb-real

fiontb.registration.icp.rgb-synthetic:
	python3 -m fiontb.registration._test.test_icp rgb-synthetic

fiontb.registration.icp.rgbd-real:
	python3 -m fiontb.registration._test.test_icp rgbd-real

fiontb.registration.icp.rgbd-synthetic:
	python3 -m fiontb.registration._test.test_icp rgbd-synthetic

fiontb.registration.icp.ms-depth-real:
	python3 -m fiontb.registration._test.test_icp ms-depth-real

fiontb.registration.icp.ms-depth-synthetic:
	python3 -m fiontb.registration._test.test_icp ms-depth-synthetic

fiontb.registration.icp.ms-rgb-real:
	python3 -m fiontb.registration._test.test_icp ms-rgb-real

fiontb.registration.icp.ms-rgb-synthetic:
	python3 -m fiontb.registration._test.test_icp ms-rgb-synthetic

fiontb.registration.icp.ms-rgbd-real:
	python3 -m fiontb.registration._test.test_icp ms-rgbd-real

fiontb.registration.icp.ms-rgbd-synthetic:
	python3 -m fiontb.registration._test.test_icp ms-rgbd-synthetic

fiontb.registration.icp.so3:
	python3 -m fiontb.registration._test.test_icp so3

fiontb.registration.icp.fail:
	python3 -m fiontb.registration._test.test_icp fail

fiontb.registration.icp.trajectory:
	python3 -m fiontb.registration._test.test_icp trajectory

fiontb.registration.autogradicp.depth-real:
	python3 -m fiontb.registration._test.test_autogradicp depth-real

fiontb.registration.autogradicp.depth-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp depth-synthetic

fiontb.registration.autogradicp.rgb-real:
	python3 -m fiontb.registration._test.test_autogradicp rgb-real

fiontb.registration.autogradicp.rgb-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp rgb-synthetic

fiontb.registration.autogradicp.rgbd-real:
	python3 -m fiontb.registration._test.test_autogradicp rgbd-real

fiontb.registration.autogradicp.rgbd-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp rgbd-synthetic

fiontb.registration.autogradicp.ms-depth-real:
	python3 -m fiontb.registration._test.test_autogradicp ms-depth-real

fiontb.registration.autogradicp.ms-depth-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp ms-depth-synthetic

fiontb.registration.autogradicp.ms-rgb-real:
	python3 -m fiontb.registration._test.test_autogradicp ms-rgb-real

fiontb.registration.autogradicp.ms-rgb-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp ms-rgb-synthetic

fiontb.registration.autogradicp.ms-rgbd-real:
	python3 -m fiontb.registration._test.test_autogradicp ms-rgbd-real

fiontb.registration.autogradicp.ms-rgbd-synthetic:
	python3 -m fiontb.registration._test.test_autogradicp ms-rgbd-synthetic

fiontb.registration.autogradicp.trajectory:
	python3 -m fiontb.registration._test.test_autogradicp trajectory

fiontb.registration.autogradicp.pcl-rgb-real:
	python3 -m fiontb.registration._test.test_autogradicp pcl-rgb-real

fiontb.registration.autogradicp.pcl-ms-rgbd-real:
	python3 -m fiontb.registration._test.test_autogradicp pcl-ms-rgbd-real

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
