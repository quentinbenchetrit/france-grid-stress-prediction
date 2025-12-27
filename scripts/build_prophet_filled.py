#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from nettoyage.data.prophet_fill import ProphetFillConfig, prophet_fill_missing  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Fill missing load values using Prophet.")
    p.add_argument("--input", default="data/processed/dataset_model_hourly.parquet")
    p.add_argument("--output", default="data/processed/dataset_model_hourly_prophetfilled.parquet")
    p.add_argument("--datetime-col", default="datetime")
    p.add_argument("--y-col", default="load_mw")
    p.add_argument("--freq", default="H", help="Regular grid frequency (default: H)")
    p.add_argument("--daily", action="store_true")
    p.add_argument("--weekly", action="store_true")
    p.add_argument("--yearly", action="store_true")
    args = p.parse_args()

    cfg = ProphetFillConfig(
        in_path=Path(args.input),
        out_path=Path(args.output),
        datetime_col=args.datetime_col,
        y_col=args.y_col,
        freq=args.freq,
        daily_seasonality=args.daily,
        weekly_seasonality=args.weekly,
        yearly_seasonality=args.yearly,
    )

    _, report = prophet_fill_missing(cfg)
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
