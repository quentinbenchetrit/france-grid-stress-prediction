"""Cleaning utilities for multi-city weather data.

The raw dataset contains hourly observations for multiple French cities.
This module standardizes the schema, checks hourly continuity per city,
and produces a national (across-city mean) hourly dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


EXPECTED_FREQ = pd.Timedelta("1H")


@dataclass
class WeatherCleanConfig:
    raw_path: Path
    out_path: Path


def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    # Handle common column names: 'date' or 'datetime'
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        raise ValueError("Missing datetime/date column")

    df = df.copy()
    df["datetime"] = dt.dt.tz_convert(None)
    for col in ["date"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize schema and basic sorting/deduplication."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "city" not in df.columns:
        raise ValueError("Missing 'city' column")

    df = _normalize_datetime(df)
    df = df.dropna(subset=["city", "datetime"]).copy()
    df["city"] = df["city"].astype(str).str.strip()

    df = df.sort_values(["city", "datetime"])
    df = df.drop_duplicates(subset=["city", "datetime"], keep="first")
    return df


def continuity_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return continuity issues per city (non-hourly steps)."""
    reports: List[Dict[str, object]] = []
    for city, dfi in df.groupby("city", sort=True):
        delta = dfi["datetime"].diff()
        n_bad_steps = int((delta.notna() & (delta != EXPECTED_FREQ)).sum())
        reports.append(
            {
                "city": city,
                "rows": int(len(dfi)),
                "min_dt": dfi["datetime"].min(),
                "max_dt": dfi["datetime"].max(),
                "n_bad_steps": n_bad_steps,
            }
        )
    return pd.DataFrame(reports).sort_values(["n_bad_steps", "city"], ascending=[False, True])


def aggregate_national_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multi-city weather to a national hourly mean."""
    # Keep numeric columns (excluding city)
    numeric_cols = [c for c in df.columns if c not in {"city", "datetime"} and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric weather variables found to aggregate")

    out = (
        df.groupby("datetime")[numeric_cols]
          .mean()
          .reset_index()
          .sort_values("datetime")
    )
    return out


def build_weather_national_dataset(cfg: WeatherCleanConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw weather CSV, clean, report continuity, export national hourly Parquet."""
    raw = pd.read_csv(cfg.raw_path)
    clean = clean_weather(raw)
    rep = continuity_report(clean)
    nat = aggregate_national_hourly(clean)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    nat.to_parquet(cfg.out_path, index=False)
    return nat, rep
