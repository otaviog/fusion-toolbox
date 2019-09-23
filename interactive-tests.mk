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

fiontb.filtering.bilateral:
	python3 -m fiontb.filtering bilateral

fiontb.filtering.featuremap:
	python3 -m fiontb.filtering featuremap

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

fiontb.pose.autogradicp.geometric:
	python3 -m fiontb.pose._test.test_autogradicp geometric --profile=True

fiontb.pose.autogradicp.color:
	python3 -m fiontb.pose._test.test_autogradicp color

fiontb.pose.autogradicp.hybrid:
	python3 -m fiontb.pose._test.test_autogradicp hybrid

fiontb.fusion.surfelfeat.merge_live:
	python3 -m fiontb.fusion.surfelfeat._test.test_merge_live
