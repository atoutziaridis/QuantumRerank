# Makefile for QuantumRerank development

.PHONY: help install install-dev test test-imports verify clean lint format docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -r requirements-test.txt
	pip install -e .

test:  ## Run all tests
	pytest tests/ -v

test-imports:  ## Test that all libraries can be imported
	pytest tests/test_imports.py -v

verify:  ## Run quantum setup verification
	python verify_quantum_setup.py

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/

lint:  ## Run code linting
	flake8 quantum_rerank/ tests/
	mypy quantum_rerank/

format:  ## Format code with black and isort
	black quantum_rerank/ tests/
	isort quantum_rerank/ tests/

docs:  ## Generate documentation
	cd docs && make html

setup-env:  ## Set up virtual environment and install dependencies
	python3 -m venv venv
	@echo "Activate the virtual environment with: source venv/bin/activate"
	@echo "Then run: make install-dev"

requirements:  ## Generate requirements.txt from current environment
	pip freeze > requirements-frozen.txt

benchmark:  ## Run performance benchmarks
	pytest tests/ -k benchmark --benchmark-only

security:  ## Run security checks
	bandit -r quantum_rerank/

pre-commit:  ## Install pre-commit hooks
	pre-commit install
	pre-commit run --all-files