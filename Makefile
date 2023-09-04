VENV?=.venv
PYTHON?=$(VENV)/bin/python3.10
PIP?=$(PYTHON) -m pip

run:
	$(PYTHON) src/main.py

venv:
	@echo "Creating venv..."
	@poetry install
	$(PIP) install --upgrade pip
	@echo "Starting the venv..."
	@poetry shell


clean:
	@echo "removing recursively: *.py[cod]"
	find . -type f -name "*.pyc" -exec rm '{}' +
	find . -type d -name "__pycache__" -exec rm -rf '{}' +
	find . -type d -name ".pytest_cache" -exec rm -rf '{}' +
	find . -type d -name "*.egg-info" -exec rm -rf '{}' +
	rm -rf $(VENV) .pybuilder
	rm -rf $(VENV)
	rm poetry.lock
	@echo "\033[31mNow, run the \`exit\` command to close the shell session created by poetry!\033[0m"'
	