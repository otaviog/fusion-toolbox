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

fiontb.datatypes:
	python3 -m unittest fiontb._tests.test_datatypes

fiontb.camera:
	python3 -m unittest fiontb._tests.test_camera

fiontb.frame:
	python3 -m unittest fiontb._tests.test_frame

fiontb.spatial.octree:
	python3 -m unittest fiontb.spatial._test.test_octtree

fiontb.spatial.coctree:
	python3 -m unittest fiontb.spatial._test.test_coctree

fiontb.spatial.benchmark_knn:
	python3 -m fiontb.spatial._test.benchmark_knn

fiontb.spatial.benchmark_knn.cprofile:
	python3 -m fiontb.spatial._test.benchmark_knn --cprofile profile.cprof

fiontb.spatial.benchmark_knn.profile:
	valgrind --tool=cachegrind python3 -m fiontb.spatial._test.benchmark_knn

fiontb.spatial.benchmark_knn_cam:
	python3 -m fiontb.spatial._test.benchmark_knn_cam 

fiontb.spatial.coctree.memcheck:
	valgrind --tool=memcheck --log-file=valgrind-out.txt python3 -m unittest fiontb.spatial._test.test_cocttree 

fiontb.spatial.aabb:
	python3 -m unittest fiontb.spatial._test.test_aabb

fiontb.spatial.indexmap:
	python3 -m unittest fiontb.spatial._test.test_indexmap

fiontb.spatial.cindexmap:
	python3 -m unittest fiontb.spatial._test.test_cindexmap

fiontb.filtering:
	python3 -m unittest fiontb._tests.test_filtering
