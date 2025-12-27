from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd


@dataclass
class ProphetFillConfig:
    in_path: Path
    out_path: Path
    datetime_col: str = "datetime"
    y_col: str = "load_mw"
    freq: str = "H"  # "H" or "30min"

    daily_seasonality: bool = True
    weekly_seasonality: bool = True
    yearly_seasonality: bool = True
    seasonality_mode: str = "additive"
    changepoint_prior_scale: float = 0.05
    interval_width: float = 0.8


def _regular_grid(df: pd.DataFrame, dt_col: str, freq: str) -> pd.DataFrame:
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(dt_col).set_index(dt_col)

    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)
    df.index.name = dt_col
    return df.reset_index()


def prophet_fill_missing(cfg: ProphetFillConfig) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Fit Prophet on observed values and fill ONLY missing target values."""
    try:
        from prophet import Prophet
    except ImportError as e:
        raise ImportError("Install Prophet: pip install prophet") from e

    if not cfg.in_path.exists():
        raise FileNotFoundError(f"Input not found: {cfg.in_path}")

    if cfg.in_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(cfg.in_path)
    elif cfg.in_path.suffix.lower() == ".csv":
        df = pd.read_csv(cfg.in_path)
    else:
        raise ValueError("Input must be .parquet or .csv")

    if cfg.datetime_col not in df.columns:
        raise KeyError(f"Missing datetime column: {cfg.datetime_col}")
    if cfg.y_col not in df.columns:
        raise KeyError(f"Missing target column: {cfg.y_col}")

    df_grid = _regular_grid(df, cfg.datetime_col, cfg.freq)

    missing_mask = df_grid[cfg.y_col].isna()
    n_missing = int(missing_mask.sum())

    df_prophet = df_grid[[cfg.datetime_col, cfg.y_col]].rename(
        columns={cfg.datetime_col: "ds", cfg.y_col: "y"}
    )
    train = df_prophet.dropna(subset=["y"]).copy()

    m = Prophet(
        daily_seasonality=cfg.daily_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        yearly_seasonality=cfg.yearly_seasonality,
        seasonality_mode=cfg.seasonality_mode,
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        interval_width=cfg.interval_width,
    )
    m.fit(train)

    future = df_prophet[["ds"]].copy()
    fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    out = df_grid.merge(fcst, left_on=cfg.datetime_col, right_on="ds", how="left").drop(columns=["ds"])

    out[f"{cfg.y_col}_original"] = out[cfg.y_col]
    out[f"{cfg.y_col}_filled"] = out[cfg.y_col]
    out.loc[missing_mask, f"{cfg.y_col}_filled"] = out.loc[missing_mask, "yhat"]

    out["filled_by_prophet"] = False
    out.loc[missing_mask, "filled_by_prophet"] = True

    out = out.rename(
        columns={
            "yhat": "prophet_yhat",
            "yhat_lower": "prophet_yhat_lower",
            "yhat_upper": "prophet_yhat_upper",
        }
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cfg.out_path, index=False)

    filled_dates = out.loc[out["filled_by_prophet"], cfg.datetime_col]
    report = {
        "status": "filled" if n_missing > 0 else "no_missing_values",
        "n_missing": n_missing,
        "filled_range": None if n_missing == 0 else (str(filled_dates.min()), str(filled_dates.max())),
        "in_path": str(cfg.in_path),
        "out_path": str(cfg.out_path),
        "freq": cfg.freq,
        "y_col": cfg.y_col,
    }
    return out, report
