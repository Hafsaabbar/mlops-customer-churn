# Makefile for Customer Churn MLOps Project (Part A)
# Compatible with GNU Make on Linux/macOS and Windows (via MSYS2 / MinGW / WSL)
# On Windows PowerShell, commands can also be run via: python scripts/run.py <target>

PYTHON ?= python

.PHONY: help install prepare validate test train gate retrain rollback fairness all status clean api test-api

help:
	@echo "Available commands:"
	@echo "  make install   : Install pinned project dependencies"
	@echo "  make prepare   : Clean raw data and create deterministic splits"
	@echo "  make validate  : Validate splits against Pandera schema"
	@echo "  make test      : Run Pytest test suite (unit and integration tests)"
	@echo "  make train     : Train baselines, Logistic Regression, Random Forest, log MLflow"
	@echo "  make gate      : Evaluate candidate with Quality Gate and promote champion"
	@echo "  make feedback  : Replay delayed offline stream feedback"
	@echo "  make retrain   : Retrain on accumulated data and gate candidate v2"
	@echo "  make rollback  : Restore previous champion model version"
	@echo "  make fairness  : Run demographic fairness audit (gender, SeniorCitizen)"
	@echo "  make all       : Full initial chain (Data -> Validation -> Train -> Gate -> Registry)"
	@echo "  make status    : Display MLflow Model Registry versions and active aliases"

install:
	$(PYTHON) -m pip install -r requirements.txt

prepare:
	$(PYTHON) -m src.data.prepare

validate:
	$(PYTHON) -m src.validation.validate

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) -m src.training.train

gate:
	$(PYTHON) -m src.lifecycle.registry promote 1

feedback:
	$(PYTHON) -m src.lifecycle.feedback feedback

retrain:
	$(PYTHON) -m src.lifecycle.feedback retrain

rollback:
	$(PYTHON) -m src.lifecycle.registry rollback

fairness:
	$(PYTHON) -m src.lifecycle.fairness

status:
	$(PYTHON) -m src.lifecycle.registry status

all: prepare validate train gate fairness status

clean:
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +

api:
	uvicorn src.api.main:app --host 127.0.0.1 --port 8000

test-api:
	python -m pytest tests/api -v



