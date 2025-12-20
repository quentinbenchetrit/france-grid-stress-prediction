"""Build the baseline modeling dataset by aligning consumption (hourly) and national weather (hourly)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


@dataclass
class MergeConfig:
    consumption_path: Path
    weather_path: Path
    out_path: Path


def build_hourly_dataset(cfg: MergeConfig) -> pd.DataFrame:
    cons = pd.read_parquet(cfg.consumption_path).copy()
    cons["datetime"] = pd.to_datetime(cons["datetime"])

    cons_hourly = (
        cons.set_index("datetime")["load_mw"]
            .resample("1H")
            .mean()
            .reset_index()
            .sort_values("datetime")
    )

    weather = pd.read_parquet(cfg.weather_path).copy()
    weather["datetime"] = pd.to_datetime(weather["datetime"])
    weather = weather.sort_values("datetime")

    final = (
        cons_hourly.merge(weather, on="datetime", how="inner")
                  .sort_values("datetime")
    )

    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(cfg.out_path, index=False)
    return final
