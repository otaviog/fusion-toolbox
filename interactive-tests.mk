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

fiontb.pose.open3d.real:
	python3 fiontb/pose/open3d_interop.py real

fiontb.pose.open3d.syn:
	python3 fiontb/pose/open3d_interop.py syn

fiontb.pose.open3d.dataset:
	python3 fiontb/pose/open3d_interop.py dataset

fiontb.pose.icp.geometric1:
	python3 -m fiontb.pose._test.test_icp geometric1

fiontb.pose.icp.color:
	python3 -m fiontb.pose._test.test_icp color

fiontb.pose.icp.multiscale-geometric:
	python3 -m fiontb.pose._test.test_icp multiscale-geometric

fiontb.pose.icp.multiscale-color:
	python3 -m fiontb.pose._test.test_icp multiscale-color

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

fiontb.fusion.surfel.merge_live:
	python3 -m fiontb.fusion.surfel._test.test_merge_live

fiontb.fusion.surfel.carve_space:
	python3 -m fiontb.fusion.surfel._test.test_carve_space

fiontb.fusion.surfel.intra_merge:
	python3 -m fiontb.fusion.surfel.intra_merge

fiontb.fusion.fsf.registration:	
	python3 -m fiontb.fusion.fsf._test.test_registration

fiontb.fusion.fsf.fusion:	
	python3 -m fiontb.fusion.fsf._test.test_fusion
