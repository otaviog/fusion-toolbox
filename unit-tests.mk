doctest:
	python doc/doctest_runner.py

fiontb.metrics:
	python3 -m unittest fiontb.metrics._test_metrics

fiontb.metrics.mesh:
	python3 -m unittest fiontb.metrics._test_metrics.TestMeshFunctions

view.fiontb.metrics.sample_points:
	python3 -m fiontb.metrics._test_metrics sample_points

fiontb.data.sens:
	python3 -m unittest fiontb.data._tests.test_sens

fiontb.data.sens.read:
	python3 -m unittest fiontb.data._tests.test_sens.TestSens.test_2_read

fiontb.data.sens.write:
	python3 -m unittest fiontb.data._tests.test_sens.TestSens.test_1_write

view.fiontb.data.sens:
	python3 -m fiontb.data._tests.test_sens

fiontb.camera:
	python3 -m unittest fiontb._tests.test_camera

fiontb.frame:
	python3 -m unittest fiontb._tests.test_frame

fiontb.spatial.aabb:
	python3 -m unittest fiontb.spatial._test.test_aabb

fiontb.spatial.trigoctree:
	python3 -m unittest fiontb.spatial._test.test_trigoctree

fiontb.filtering:
	python3 -m unittest fiontb._tests.test_filtering

fiontb.filtering.featuremap:
	python3 -m unittest fiontb._tests.test_filtering.TestFiltering.test_featuremap

fiontb.pose.so3:
	python3 -m unittest fiontb.pose._test.test_so3
