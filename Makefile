LINT_PATHS=gym_space_engineers/ setup.py tests

pytest:
	python3 -m pytest -v

type:
	pytype -j auto

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://lintlyci.github.io/Flake8Rules/
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings.
	flake8 ${LINT_PATHS} --count --exit-zero --statistics

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 127 ${LINT_PATHS}

diff-format:
	# Reformat only lines that were changed
	darker --isort -l 127 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black -l 127 --check ${LINT_PATHS}

commit-checks: format type lint

.PHONY: clean spelling doc lint format check-codestyle commit-checks
