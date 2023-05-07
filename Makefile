VENV?=venv
PYTHON?=$(VENV)/bin/python3.10
PIP?=$(PYTHON) -m pip

help:
	@echo "to use it:"
	@echo "   1 - Run: make venv"
	@echo "   2 - Run: make run"

venv:$(VENV)/bin/activate
$(VENV)/bin/activate: requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

run:
	source $(VENV)/bin/activate
	$(PYTHON) src/main.py

clean:
	@rm -rf $(VENV)