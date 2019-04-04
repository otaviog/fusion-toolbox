about:
	@echo "Project maintaining tasks."

doc-create:
	rm -f doc/source/rflow.*.rst
	sphinx-apidoc -o doc/source rflow
	make -C doc/ html

doc-open:
	sensible-browser doc/build/html/index.html

pylint:
	python3 -m pylint rflow

pep8:
	python3 -m autopep8 --recursive --in-place rflow

local-ci.pages:
	gitlab-ci-multi-runner exec docker pages\
		--docker-pull-policy=never

local-ci:
	gitlab-ci-multi-runner exec docker test\
		--docker-pull-policy=never

ut-metrics:
	python3 -m unittest fiontb.metrics._test_metrics

ut-datatypes:
	python3 -m unittest fiontb._tests.test_datatypes
