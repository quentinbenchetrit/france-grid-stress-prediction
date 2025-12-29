PYTHON := python

# ---------- Targets ----------
.PHONY: help data features train backtest eda clean

help:
	@echo "Available commands:"
	@echo "  make data      -> build cleaned CSV datasets"
	@echo "  make features  -> build feature datasets"
	@echo "  make train     -> train load forecasting model"
	@echo "  make backtest  -> run backtesting"
	@echo "  make eda       -> open EDA notebooks"
	@echo "  make clean     -> remove generated files"

# ---------- Data ----------
data:
	$(PYTHON) scripts/make_dataset.py

# ---------- Feature engineering ----------
features:
	$(PYTHON) src/fgsp/features/build_lags.py
	$(PYTHON) src/fgsp/features/calendar_features.py

# ---------- Modeling ----------
train:
	$(PYTHON) scripts/train_load.py

backtest:
	$(PYTHON) scripts/backtest_load.py

# ---------- EDA ----------
eda:
	jupyter notebook notebooks/

# ---------- Cleaning ----------
clean:
	rm -rf data/interim/*
	rm -rf data/processed/*
	rm -rf models/*
	rm -rf _pycache_
	find . -name "*.pyc" -delete