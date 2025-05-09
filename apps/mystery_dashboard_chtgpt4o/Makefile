# Makefile for Mystery Dashboard
VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: env install run

env:
	python3 -m venv $(VENV)

install: env
	$(PIP) install -U preswald
	$(PIP) install .

run: install
	$(PYTHON) run.py