about:
	@echo "Project maintaining tasks."

pylint:
	python3 -m pylint rflow

pep8:
	python3 -m autopep8 --recursive --in-place rflow

doc-create:
	rm -f doc/source/slamtb.*
	sphinx-apidoc -o doc/source slamtb
	make -C doc/ html

doc-open:
	sensible-browser doc/build/html/index.html

clang-format:
	clang-format -i **/*.cpp **/*.hpp

cpp-doc-create:
	doxygen Doxyfile

cpp-doc-open:
	sensible-browser doc/cpp/html/index.html

local-ci.pages:
	gitlab-ci-multi-runner exec docker pages\
		--docker-pull-policy=never

local-ci:
	gitlab-ci-multi-runner exec docker test\
		--docker-pull-policy=never

