about:
	@echo "Empirical test tasks"

#########
# Filtering
#

fiontb.filtering.bilateral:
	python3 -m fiontb.filtering bilateral

fiontb.filtering.featuremap:
	python3 -m fiontb.filtering featuremap

fiontb.spatial.indexmap:
	python3 -m fiontb.spatial.indexmap

#########
# Pose
#

fiontb.pose.open3d.rgbd-real:
	python3 -m fiontb.pose.open3d_interop rgbd-real

fiontb.pose.open3d.rgbd-synthetic:
	python3 -m fiontb.pose.open3d_interop rgbd-synthetic

fiontb.pose.open3d.coloricp-real:
	python3 -m fiontb.pose.open3d_interop coloricp-real

fiontb.pose.open3d.coloricp-synthetic:
	python3 -m fiontb.pose.open3d_interop coloricp-synthetic

fiontb.pose.icp.geometric1:
	python3 -m fiontb.pose._test.test_icp geometric1

fiontb.pose.icp.color:
	python3 -m fiontb.pose._test.test_icp color

fiontb.pose.icp.hybrid:
	python3 -m fiontb.pose._test.test_icp hybrid

fiontb.pose.icp.fail:
	python3 -m fiontb.pose._test.test_icp fail

fiontb.pose.icp.multiscale-geometric:
	python3 -m fiontb.pose._test.test_icp multiscale-geometric

fiontb.pose.icp.multiscale-hybrid:
	python3 -m fiontb.pose._test.test_icp multiscale-hybrid

fiontb.pose.autogradicp.geometric:
	python3 -m fiontb.pose._test.test_autogradicp geometric --profile=True

fiontb.pose.autogradicp.color:
	python3 -m fiontb.pose._test.test_autogradicp color

fiontb.pose.autogradicp.hybrid:
	python3 -m fiontb.pose._test.test_autogradicp hybrid

fiontb.pose.autogradicp.multiscale-geometric:
	python3 -m fiontb.pose._test.test_autogradicp multiscale-geometric

fiontb.pose.autogradicp.multiscale-hybrid:
	python3 -m fiontb.pose._test.test_autogradicp multiscale-hybrid

#########
# Surfel
#

fiontb.surfel:
	python3 -m fiontb._test.test_surfel adding

fiontb.viz.surfelrender:
	python3 -m fiontb.viz.surfelrender

fiontb.fusion.surfel.indexmap.surfel-raster:
	python3 -m fiontb.fusion.surfel._test.test_indexmap surfel-raster

fiontb.fusion.surfel.merge_live:
	python3 -m fiontb.fusion.surfel._test.test_merge_live

fiontb.fusion.surfel.merge:
	python3 -m fiontb.fusion.surfel._test.test_merge

fiontb.fusion.surfel.carve_space:
	python3 -m fiontb.fusion.surfel._test.test_carve_space

fiontb.fusion.surfel.remove_unstable:
	python3 -m fiontb.fusion.surfel._test.test_remove_unstable

fiontb.fusion.surfel.fusion:
	python3 -m fiontb.fusion.surfel._test.test_fusion

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

fiontb.pipeline.fsf.real-scene:
	python3 -m fiontb.pipeline._test.test_fsf_slam real_scene

fiontb.pipeline.fsf.synthetic-scene:
	python3 -m fiontb.pipeline._test.test_fsf_slam synthetic_scene
