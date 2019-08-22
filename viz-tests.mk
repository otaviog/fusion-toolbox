about:
	@echo "Empirical test tasks"

fiontb.fusion.surfel.spacecarving:
	python3 -m fiontb.fusion.surfel.spacecarving

fiontb.fusion.surfel.live_merge:
	python3 -m fiontb.fusion.surfel.live_merge

fiontb.fusion.surfel.intra_merge:
	python3 -m fiontb.fusion.surfel.intra_merge

fiontb.viz.surfelrender:
	python3 -m fiontb.viz.surfelrender

fiontb.filtering:
	python3 -m fiontb.filtering

fiontb.spatial.indexmap:
	python3 -m fiontb.spatial.indexmap

fiontb.pose.open3d:
	python3 -m fiontb.pose.open3d

fiontb.pose.icp.geometric1:
	python3 -m fiontb.pose._test.test_icp geometric1

fiontb.pose.icp.geometric2:
	python3 -m fiontb.pose._test.test_icp geometric2

fiontb.pose.icp.multiscale-geometric:
	python3 -m fiontb.pose._test.test_icp multiscale-geometric

fiontb.pose.icp.intensity:
	python3 -m fiontb.pose._test.test_icp intensity

