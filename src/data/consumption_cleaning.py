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
    """Load and clean one raw consumption CSV (Robust version)."""
    
    # 1. Lecture robuste (détection auto du séparateur : , ou ;)
    try:
        df = pd.read_csv(path, sep=None, engine='python')
    except Exception:
        # Si échec, on tente le point-virgule explicitement
        df = pd.read_csv(path, sep=';')

    # 2. Standardisation des noms de colonnes (minuscules + strip)
    df.columns = [c.strip().lower() for c in df.columns]

    # 3. Dictionnaire de renommage (Français -> Format attendu)
    # Adapte cette liste si tes colonnes s'appellent autrement (ex: "date_heure")
    rename_map = {
        "date": "datetime",
        "date_heure": "datetime",
        "horodatage": "datetime",
        "consommation": "load_mw",
        "puissance": "load_mw",
        "valeur": "load_mw",
        "value": "load_mw"
    }
    df = df.rename(columns=rename_map)

    # 4. Vérification de la colonne datetime
    if "datetime" not in df.columns:
        # Si on a "date" et "heure" séparés, on tente de les fusionner
        if "date" in df.columns and "heure" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["heure"])
        else:
            raise ValueError(f"Colonne 'datetime' introuvable dans {path.name}. Colonnes dispos: {list(df.columns)}")

    # 5. Conversion et nettoyage
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    
    # Retrait timezone pour éviter les conflits
    df["datetime"] = df["datetime"].dt.tz_convert(None)

    # 6. Gestion de la colonne charge (load_mw)
    if "load_mw" not in df.columns:
        # On cherche une colonne numérique restante
        numeric_cols = [c for c in df.columns if c != "datetime" and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
             df = df.rename(columns={numeric_cols[0]: "load_mw"})
        else:
            raise ValueError(f"Pas de colonne de consommation trouvée dans {path.name}")

    # On ne garde que l'essentiel
    df = df[["datetime", "load_mw"]].copy()
    
    # Dédoublonnage (garde la première occurrence)
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
