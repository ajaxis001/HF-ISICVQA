# 'install' target is used to set up the environment. This will first upgrade pip (Python's package installer) and then install the dependencies listed in the requirements.txt file.
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

# 'test' target will run the unit tests using the pytest module on all Python files within the 'tests' directory. The '-vv' flag is for verbose output, and '--cov' is for measuring code coverage.
test:
	python -m pytest -vv --cov=app tests/*.py

# 'lint' target will run the linter (pylint) on all Python files in the current directory, while disabling warning categories R (Refactor) and C (Convention).
lint:
	pylint --disable=R,C *.py

# 'format' target will format all Python files in the current directory to meet the PEP 8 style guide using the Black code formatter.
format:
	black *.py

# 'deploy' target will be used for deployment. The actual commands aren't provided here.
deploy:
	# deploy goes here

# 'all' target runs the install, lint, test, and format targets. This is a sort of "meta" target that allows you to run multiple targets with one command.
all: install lint test format
