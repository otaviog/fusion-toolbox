doctest:
	python doc/doctest_runner.py

fiontb.metrics:
	python3 -m unittest fiontb.metrics._test_metrics

fiontb.data.sens:
	python3 -m unittest fiontb.data._tests.test_sens

fiontb.data.sens.read:
	python3 -m unittest fiontb.data._tests.test_sens.TestSens.test_2_read

fiontb.data.sens.write:
	python3 -m unittest fiontb.data._tests.test_sens.TestSens.test_1_write

fiontb.data.sens.view:
	python3 -m fiontb.data._tests.test_sens
