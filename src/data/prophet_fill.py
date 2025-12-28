"""
Prophet-based imputation for time series (fills missing values using yhat).

Usage (CLI):
    python prophet_fill.py \
        --input /path/to/input.parquet \
        --output /path/to/output.parquet \
        --date-col datetime \
        --value-col load_mw \
        --freq H

Notes:
- The script trains Prophet on non-missing rows, predicts on the full timeline,
  and fills only the missing values of `value-col` with Prophet's yhat.
- Output keeps the original columns and adds:
    - filled_by_prophet (bool): True where the value was imputed.

Requirements:
    pip install prophet pandas pyarrow
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

try:
    from prophet import Prophet
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Cannot import Prophet. Install with: pip install prophet"
    ) from e


def build_prophet_model(
    daily_seasonality: bool = True,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
    seasonality_mode: str = "additive",
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """Create a Prophet model with sensible defaults for hourly load-like series."""
    return Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )


def fill_missing_with_prophet(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str | None = None,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Fit Prophet on non-missing (date_col, value_col) rows and impute missing values.

    If `freq` is provided, the function will reindex to a complete date range before
    predicting, then merge back to the original rows. If not provided, it predicts
    only on the timestamps present in df.
    """
    model_kwargs = model_kwargs or {}

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])

    # Track missing in the ORIGINAL dataframe (before any reindexing)
    missing_mask = out[value_col].isna()

    # Training set
    train = out.loc[~missing_mask, [date_col, value_col]].rename(
        columns={date_col: "ds", value_col: "y"}
    )
    if train.empty:
        raise ValueError("No non-missing rows available to train Prophet.")

    m = build_prophet_model(**model_kwargs)
    m.fit(train)

    if freq:
        # Build complete timeline between min/max (useful if timestamps are regular)
        ds_full = pd.date_range(out[date_col].min(), out[date_col].max(), freq=freq)
        future = pd.DataFrame({"ds": ds_full})
    else:
        future = pd.DataFrame({"ds": out[date_col].sort_values().unique()})

    forecast = m.predict(future)[["ds", "yhat"]]

    # Merge forecast back to original rows
    out = out.merge(
        forecast.rename(columns={"ds": date_col}),
        on=date_col,
        how="left",
        validate="many_to_one",
    )

    # Fill ONLY missing original values
    out["filled_by_prophet"] = missing_mask
    out.loc[missing_mask, value_col] = out.loc[missing_mask, "yhat"]

    return out.drop(columns=["yhat"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill missing values using Prophet.")
    p.add_argument("--input", required=True, type=Path, help="Input parquet/csv file.")
    p.add_argument("--output", required=True, type=Path, help="Output parquet/csv file.")
    p.add_argument("--date-col", default="datetime", help="Datetime column name.")
    p.add_argument("--value-col", default="load_mw", help="Target value column name.")
    p.add_argument("--freq", default=None, help="Pandas frequency (e.g., H, 15min). Optional.")
    p.add_argument("--format", default=None, choices=["parquet", "csv"], help="Force I/O format.")
    # A few Prophet knobs (optional)
    p.add_argument("--seasonality-mode", default="additive", choices=["additive", "multiplicative"])
    p.add_argument("--changepoint-prior-scale", default=0.05, type=float)
    p.add_argument("--no-daily", action="store_true")
    p.add_argument("--no-weekly", action="store_true")
    p.add_argument("--yearly", action="store_true")
    return p.parse_args()


def read_df(path: Path, fmt: str | None) -> pd.DataFrame:
    fmt = fmt or path.suffix.lower().lstrip(".")
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {fmt}. Use parquet or csv, or pass --format.")


def write_df(df: pd.DataFrame, path: Path, fmt: str | None) -> None:
    fmt = fmt or path.suffix.lower().lstrip(".")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
        return
    if fmt == "csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {fmt}. Use parquet or csv, or pass --format.")


def main() -> None:
    args = parse_args()

    df = read_df(args.input, args.format)

    model_kwargs = {
        "daily_seasonality": not args.no_daily,
        "weekly_seasonality": not args.no_weekly,
        "yearly_seasonality": args.yearly,
        "seasonality_mode": args.seasonality_mode,
        "changepoint_prior_scale": args.changepoint_prior_scale,
    }

    df_filled = fill_missing_with_prophet(
        df=df,
        date_col=args.date_col,
        value_col=args.value_col,
        freq=args.freq,
        model_kwargs=model_kwargs,
    )

    write_df(df_filled, args.output, args.format)

    n_missing = df[args.value_col].isna().sum()
    print(f"Done. Filled {n_missing} missing values (only where {args.value_col} was NaN).")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
