# =====================================
# Project: France Grid Stress Prediction
# =====================================

.PHONY: help venv install dataset train-load backtest clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make venv           Create virtual environment"
	@echo "  make install        Install project in editable mode"
	@echo "  make dataset        Build processed dataset"
	@echo "  make train-load     Train load forecasting model"
	@echo "  make backtest       Run day-ahead backtest"
	@echo "  make clean          Remove caches and temporary files"

# Create virtual environment
venv:
	python -m venv .venv
	@echo "Virtual environment created. Activate it with:"
	@echo "source .venv/bin/activate"

# Install dependencies and project
install:
	pip install --upgrade pip
	pip install -e .

# Build dataset
dataset:
	python scripts/make_dataset.py --config configs/data.yaml

# Train load forecasting model
train-load:
	python scripts/train_load.py --config configs/model_load.yaml

# Run day-ahead backtest
backtest:
	python scripts/backtest_load.py --config configs/model_load.yaml

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf .pytest_cache
