# File: src/data/prophet_fill.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd


@dataclass
class ProphetFillConfig:
    """
    Configuration for Prophet-based counterfactual filling of missing consumption values.

    This is meant for methodological comparison only (not production imputation).
    """
    in_path: Path
    out_path: Path
    datetime_col: str = "datetime"
    y_col: str = "load_mw"  # adapt if your baseline uses another name
    freq: str = "H"         # hourly baseline expected
    # Prophet settings (keep minimal; you can extend later)
    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    interval_width: float = 0.8


def _ensure_hourly_index(df: pd.DataFrame, dt_col: str, freq: str) -> pd.DataFrame:
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col])
    out = out.sort_values(dt_col)
    out = out.set_index(dt_col)

    # Reindex to expected regular grid (hourly)
    full_index = pd.date_range(start=out.index.min(), end=out.index.max(), freq=freq)
    out = out.reindex(full_index)
    out.index.name = dt_col
    return out.reset_index()


def prophet_fill_missing(cfg: ProphetFillConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Load baseline dataset (hourly), fit Prophet on observed values, and fill only missing y values.

    Returns:
        df_filled: dataset with column `filled_by_prophet` added, and y filled where missing
        report: basic diagnostics
    """
    # Lazy import so the rest of the project works without Prophet installed
    try:
        from prophet import Prophet
    except ImportError as e:
        raise ImportError(
            "Prophet is not installed. Install it (e.g. `pip install prophet`) "
            "or run this step in an environment where prophet is available."
        ) from e

    if not Path(cfg.in_path).exists():
        raise FileNotFoundError(f"Input baseline dataset not found: {cfg.in_path}")

    df = pd.read_parquet(cfg.in_path)
    if cfg.datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column '{cfg.datetime_col}' in {cfg.in_path}")
    if cfg.y_col not in df.columns:
        raise KeyError(f"Missing target column '{cfg.y_col}' in {cfg.in_path}")

    # Ensure regular hourly grid (helps Prophet and makes missingness explicit)
    df_grid = _ensure_hourly_index(df, cfg.datetime_col, cfg.freq)

    # Identify missing y
    missing_mask = df_grid[cfg.y_col].isna()
    n_missing = int(missing_mask.sum())
    if n_missing == 0:
        # still write output with flag
        out = df_grid.copy()
        out["filled_by_prophet"] = False
        cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(cfg.out_path, index=False)
        return out, {
            "status": "no_missing_values",
            "n_missing": 0,
            "filled_range": None,
            "in_path": str(cfg.in_path),
            "out_path": str(cfg.out_path),
        }

    # Prepare Prophet frame using observed values only
    df_prophet = df_grid[[cfg.datetime_col, cfg.y_col]].rename(
        columns={cfg.datetime_col: "ds", cfg.y_col: "y"}
    )

    train = df_prophet[~df_prophet["y"].isna()].copy()

    m = Prophet(
        daily_seasonality=cfg.daily_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        yearly_seasonality=cfg.yearly_seasonality,
        seasonality_mode=cfg.seasonality_mode,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        interval_width=cfg.interval_width,
    )
    m.fit(train)

    # Forecast on full grid
    future = df_prophet[["ds"]].copy()
    fcst = m.predict(future)[["ds", "yhat"]]

    # Fill only where missing
    out = df_grid.copy()
    out = out.merge(fcst, left_on=cfg.datetime_col, right_on="ds", how="left").drop(columns=["ds"])
    out["filled_by_prophet"] = False
    out.loc[missing_mask, cfg.y_col] = out.loc[missing_mask, "yhat"]
    out.loc[missing_mask, "filled_by_prophet"] = True
    out = out.drop(columns=["yhat"])

    # Report
    missing_dates = out.loc[out["filled_by_prophet"], cfg.datetime_col]
    filled_range = (missing_dates.min(), missing_dates.max())

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.out_path, index=False)

    report = {
        "status": "filled",
        "n_missing": n_missing,
        "filled_range": (str(filled_range[0]), str(filled_range[1])),
        "in_path": str(cfg.in_path),
        "out_path": str(cfg.out_path),
        "y_col": cfg.y_col,
        "freq": cfg.freq,
    }
    return out, report
