about:
	@echo "Shortcuts to run unit tests"

fiontb.metrics.trajectory:
	python3 -m unittest fiontb.metrics._test.test_metrics.TestTrajectoryMetrics

fiontb.metrics.geometry:
	python3 -m unittest fiontb.metrics._test.test_metrics.TestGeometryMetrics

fiontb.metrics.mesh:
	python3 -m unittest fiontb.metrics._test_metrics.TestMeshFunctions

fiontb.data.sens:
	python3 -m unittest fiontb.data._test.test_sens

fiontb.data.sens.read:
	python3 -m unittest fiontb.data._test.test_sens.TestSens.test_2_read

fiontb.data.sens.write:
	python3 -m unittest fiontb.data._test.test_sens.TestSens.test_1_write

view.fiontb.data.sens:
	python3 -m fiontb.data._test.test_sens

fiontb.camera:
	python3 -m unittest fiontb._test.test_camera

fiontb.processing:
	python3 -m unittest fiontb._test.test_processing

fiontb.spatial.trigoctree:
	python3 -m unittest fiontb.spatial._test.test_trigoctree

fiontb.filtering:
	python3 -m unittest fiontb._test.test_filtering

fiontb.filtering.featuremap:
	python3 -m unittest fiontb._test.test_filtering.TestFiltering.test_featuremap

fiontb.pose.so3:
	python3 -m unittest fiontb.pose._test.test_so3

fiontb.spatial.kdtreelayer:
	python3 -m unittest fiontb.spatial._test.test_kdtree_layer

fiontb.spatial.fpcl_matcher:
	python3 -m unittest fiontb.spatial._test.test_fpcl_matcher
