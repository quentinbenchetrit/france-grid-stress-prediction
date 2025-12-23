"""Cleaning utilities for multi-city weather data (multi-files).

The raw dataset contains hourly observations for multiple French cities.
This module standardizes the schema, checks hourly continuity per city,
and produces a national (across-city mean) hourly dataset.

This version supports multiple yearly CSV files (e.g. 2010–2024).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd


EXPECTED_FREQ = pd.Timedelta("1H")


@dataclass
class WeatherCleanConfig:
    raw_dir: Path
    out_path: Path
    pattern: str = "weather_32_cities*.csv"  # picks historical_2010..2014 + 2015..2024 by default


def _normalize_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=True)
    else:
        raise ValueError("Missing datetime/date column")

    df = df.copy()
    df["datetime"] = dt.dt.tz_convert(None)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    return df


def clean_weather(df: pd.DataFrame) -> pd.DataFrame:
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
    numeric_cols = [
        c for c in df.columns
        if c not in {"city", "datetime"} and pd.api.types.is_numeric_dtype(df[c])
    ]
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
    files = sorted(cfg.raw_dir.glob(cfg.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {cfg.pattern} in {cfg.raw_dir}")

    frames: List[pd.DataFrame] = []
    for p in files:
        raw = pd.read_csv(p)
        frames.append(raw)

    raw_all = pd.concat(frames, ignore_index=True)

    clean = clean_weather(raw_all)
    rep = continuity_report(clean)
    nat = aggregate_national_hourly(clean)

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    nat.to_parquet(cfg.out_path, index=False)
    return nat, rep
