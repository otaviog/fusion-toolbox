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

fiontb.pose.open3d.real:
	python3 fiontb/pose/open3d_interop.py real

fiontb.pose.open3d.syn:
	python3 fiontb/pose/open3d_interop.py syn

fiontb.pose.open3d.dataset:
	python3 fiontb/pose/open3d_interop.py dataset

fiontb.pose.icp.geometric1:
	python3 -m fiontb.pose._test.test_icp geometric1

fiontb.pose.icp.geometric2:
	python3 -m fiontb.pose._test.test_icp geometric2

fiontb.pose.icp.multiscale-geometric:
	python3 -m fiontb.pose._test.test_icp multiscale-geometric

fiontb.pose.icp.intensity:
	python3 -m fiontb.pose._test.test_icp intensity


fiontb.pose.operators1:
	python3 -m fiontb.pose.operators projection

fiontb.pose.operators2:
	python3 -m fiontb.pose.operators full_so3
