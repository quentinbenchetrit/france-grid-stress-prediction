"""Cleaning utilities for French national electricity consumption data.

This module standardizes and cleans `consommation_YYYY_long.csv` files and
exports a consolidated Parquet dataset.

Assumptions
-----------
- Time resolution in raw files is 30 minutes.
- `datetime` is parsable by pandas.
- A `statut` column may exist and is removed (quality flag handled upstream).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional

import pandas as pd


EXPECTED_FREQ = pd.Timedelta("30min")


@dataclass
class ConsumptionCleanConfig:
    raw_dir: Path
    out_path: Path
    pattern: str = "consommation_*_long.csv"


def clean_consumption_file(path: Path) -> pd.DataFrame:
    """Load and clean one raw consumption CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "datetime" not in df.columns:
        raise ValueError(f"Missing 'datetime' in {path.name}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime")

    # Drop non-essential columns if present
    for col in ["statut", "slot_index"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only numeric load column; attempt common names
    load_candidates = [c for c in df.columns if c in {"load_mw", "consumption_mw", "conso_mw", "valeur"}]
    if "load_mw" not in df.columns:
        if load_candidates:
            df = df.rename(columns={load_candidates[0]: "load_mw"})
        else:
            # Fallback: first numeric column besides datetime
            numeric_cols = [c for c in df.columns if c != "datetime" and pd.api.types.is_numeric_dtype(df[c])]
            if not numeric_cols:
                raise ValueError(f"No numeric load column found in {path.name}")
            df = df.rename(columns={numeric_cols[0]: "load_mw"})

    df = df[["datetime", "load_mw"]].copy()

    # Deduplicate timestamps (keep first)
    df = df.drop_duplicates(subset=["datetime"], keep="first")

    return df


def build_consumption_dataset(cfg: ConsumptionCleanConfig) -> pd.DataFrame:
    """Clean all matching raw files and export consolidated Parquet."""
    files = sorted(cfg.raw_dir.glob(cfg.pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {cfg.pattern} in {cfg.raw_dir}")

    cleaned: List[pd.DataFrame] = []
    reports: List[Dict[str, object]] = []

    for p in files:
        df = clean_consumption_file(p)

        # simple continuity check at 30-min resolution
        delta = df["datetime"].diff()
        n_bad_steps = int((delta.notna() & (delta != EXPECTED_FREQ)).sum())

        reports.append(
            {
                "file": p.name,
                "rows": int(len(df)),
                "min_dt": df["datetime"].min(),
                "max_dt": df["datetime"].max(),
                "n_bad_steps": n_bad_steps,
            }
        )
        cleaned.append(df)

    df_all = pd.concat(cleaned, ignore_index=True).sort_values("datetime")
    df_all = df_all.drop_duplicates(subset=["datetime"], keep="first")

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(cfg.out_path, index=False)

    return df_all, pd.DataFrame(reports).sort_values("file")
