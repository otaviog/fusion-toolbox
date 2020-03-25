about:
	@echo "Shortcuts to run unit tests"

slamtb.metrics.trajectory:
	python3 -m unittest slamtb.metrics._test.test_metrics.TestTrajectoryMetrics

slamtb.metrics.geometry:
	python3 -m unittest slamtb.metrics._test.test_metrics.TestGeometryMetrics

slamtb.metrics.mesh:
	python3 -m unittest slamtb.metrics._test_metrics.TestMeshFunctions

slamtb.camera:
	python3 -m unittest slamtb._test.test_camera

slamtb.processing:
	python3 -m unittest slamtb._test.test_processing

slamtb.spatial.trigoctree:
	python3 -m unittest slamtb.spatial._test.test_trigoctree

slamtb.registration.se3:
	python3 -m unittest slamtb.registration._test.test_se3

slamtb.spatial.kdtreelayer:
	python3 -m unittest slamtb.spatial._test.test_kdtree_layer

slamtb.spatial.fpcl_matcher:
	python3 -m unittest slamtb.spatial._test.test_fpcl_matcher
