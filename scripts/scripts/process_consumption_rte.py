#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_year_raw(raw_cons_dir: Path, year: int) -> pd.DataFrame | None:
    """
    Load RTE consumption for a given year from:
      data/raw/consommation/<year>/Historique_consommation_INST_<year>.xls[x]|csv|parquet

    Your current structure example:
      /home/onyxia/work/france-grid-stress-prediction/data/raw/consommation/1996/Historique_consommation_INST_1996.xls
    """
    year_dir = raw_cons_dir / str(year)
    if not year_dir.exists():
        print(f"[{year}] Missing folder: {year_dir}")
        return None

    # Prefer the canonical filename first (avoids picking random files in the folder)
    preferred = []
    for ext in [".parquet", ".csv", ".xlsx", ".xls"]:
        preferred.append(year_dir / f"Historique_consommation_INST_{year}{ext}")

    for p in preferred:
        if p.exists():
            path = p
            break
    else:
        # fallback: take the first file in folder with known extensions
        candidates = []
        for ext in ("*.parquet", "*.csv", "*.xlsx", "*.xls"):
            candidates += list(year_dir.glob(ext))
        if not candidates:
            print(f"[{year}] No readable file found in {year_dir}")
            return None
        # prefer csv > xls(x) > parquet only if parquet not intended; here parquet is ok if present
        def score(p: Path) -> int:
            s = p.suffix.lower()
            if s == ".parquet": return 0
            if s == ".csv": return 1
            if s in (".xlsx", ".xls"): return 2
            return 9
        path = sorted(candidates, key=score)[0]

    print(f"[{year}] Reading: {path.name}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";", engine="python")

    if path.suffix.lower() in (".xls", ".xlsx"):
        # xlrd>=2.0.1 should be installed for xls
        return pd.read_excel(path)

    print(f"[{year}] Unsupported format: {path.name}")
    return None


def tidy_year(df_raw: pd.DataFrame, year: int, step_minutes: int = 30) -> pd.DataFrame:
    """
    Convert RTE daily wide format to long format.
    Output: datetime, date, year, statut, slot_index, load_mw
    """
    df = df_raw.copy()
    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # Parse day date from first column
    df["date"] = pd.to_datetime(df["col_0"], format="%d/%m/%Y", errors="coerce")

    # Keep only valid-date rows (drops headers/empty rows)
    df = df[df["date"].notna()].copy()

    # Drop fully-empty columns early (many early XLS have empty col_1)
    df = df.dropna(axis=1, how="all")

    # Create "statut" safely
    if "col_1" in df.columns:
        df["statut"] = df["col_1"]
    else:
        df["statut"] = None

    # Time columns = everything except date/statut/col_0/col_1
    drop_cols = {"date", "statut", "col_0", "col_1"}
    time_cols = [c for c in df.columns if c not in drop_cols]

    if not time_cols:
        raise ValueError(f"[{year}] No time-step columns detected after cleaning.")

    # Melt to long
    df_long = df.melt(
        id_vars=["date", "statut"],
        value_vars=time_cols,
        var_name="slot_col",
        value_name="load_mw",
    )

    df_long = df_long.sort_values(["date", "slot_col"]).reset_index(drop=True)

    df_long["slot_index"] = df_long.groupby("date").cumcount()
    df_long["datetime"] = df_long["date"] + pd.to_timedelta(df_long["slot_index"] * step_minutes, unit="m")

    df_long["year"] = year
    df_long = df_long[["datetime", "date", "year", "statut", "slot_index", "load_mw"]]

    return df_long

def process_year(raw_cons_dir: Path, out_dir: Path, year: int, step_minutes: int = 30) -> None:
    df_raw = load_year_raw(raw_cons_dir, year)
    if df_raw is None:
        print(f"[{year}] Skipped (no raw data).")
        return

    df_long = tidy_year(df_raw, year, step_minutes=step_minutes)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"consommation_{year}_long.csv"
    df_long.to_csv(out_path, index=False)

    print(f"[{year}] Saved: {out_path} ({df_long.shape[0]} rows)")


def parse_years(years: str) -> list[int]:
    years = years.strip()
    if "-" in years:
        a, b = years.split("-", 1)
        start = int(a)
        end = int(b)
        if end < start:
            raise ValueError("Invalid year range.")
        return list(range(start, end + 1))
    return [int(y.strip()) for y in years.split(",") if y.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert RTE consumption yearly files (xls/csv/parquet) into long CSV format."
    )
    p.add_argument(
        "--base-dir",
        default="/home/onyxia/work/france-grid-stress-prediction",
        help="Project root (default matches your Onyxia workspace).",
    )
    p.add_argument(
        "--raw-cons-dir",
        default="data/raw/consommation",
        help="Raw consumption folder under base-dir.",
    )
    p.add_argument(
        "--out-dir",
        default="data/interim/consommation",
        help="Output folder under base-dir (use interim for EDA-friendly CSV).",
    )
    p.add_argument(
        "--years",
        default="1996-2025",
        help='Example: "1996-2025" or "2020,2021,2024".',
    )
    p.add_argument(
        "--step-minutes",
        type=int,
        default=30,
        help="Time step in minutes (default: 30).",
    )
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_cons_dir = (base_dir / args.raw_cons_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()

    years = parse_years(args.years)

    print(f"Base dir       : {base_dir}")
    print(f"Raw cons dir   : {raw_cons_dir}")
    print(f"Out dir        : {out_dir}")
    print(f"Years          : {years[0]}..{years[-1]} ({len(years)} years)")
    print(f"Step (minutes) : {args.step_minutes}")

    for y in years:
        try:
            process_year(raw_cons_dir, out_dir, y, step_minutes=args.step_minutes)
        except Exception as e:
            print(f"[{y}] ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
