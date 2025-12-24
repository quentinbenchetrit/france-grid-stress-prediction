#!/usr/bin/env python3
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pandas as pd




def load_year_raw(raw_cons_dir: Path, year: int) -> pd.DataFrame | None:
    """
    Charge les données RTE pour une année donnée.
    Supporte une arborescence data/raw/consommation/<year>/.
    """
    year_dir = raw_cons_dir / str(year)
    if not year_dir.exists():
        print(f"[{year}] Dossier manquant : {year_dir}")
        return None

    # fichiers possibles
    candidates = list(year_dir.glob("*"))

    if not candidates:
        print(f"[{year}] Aucun fichier dans {year_dir}")
        return None

    # priorité aux formats lisibles
    def score(p: Path) -> int:
        s = p.suffix.lower()
        if s == ".parquet":
            return 0
        if s == ".csv":
            return 1
        if s in (".xlsx", ".xls"):
            return 2
        if s == ".zip":
            return 3
        return 9

    candidates = sorted(candidates, key=score)
    path = candidates[0]

    print(f"[{year}] Lecture : {path.name}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";", engine="python")

    if path.suffix in (".xls", ".xlsx"):
        return pd.read_excel(path)

    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            inner = sorted(z.namelist())[0]
            print(f"[{year}] ZIP → {inner}")
            with z.open(inner) as f:
                if inner.lower().endswith(".csv"):
                    try:
                        return pd.read_csv(f)
                    except Exception:
                        return pd.read_csv(f, sep=";", engine="python")
                if inner.lower().endswith((".xls", ".xlsx")):
                    return pd.read_excel(f)

    print(f"[{year}] Format non supporté : {path.name}")
    return None



def tidy_year(df_raw: pd.DataFrame, year: int, step_minutes: int = 30) -> pd.DataFrame:
    """
    Transforme le format 'large' journalier RTE en format 'long' :
    une ligne = un pas de temps (par défaut : 30 minutes).
    Colonnes finales : datetime, date, year, statut, slot_index, load_mw
    """
    df = df_raw.copy()
    df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # col_0 = dates 'dd/mm/YYYY' (ou NaN / texte)
    df["date"] = pd.to_datetime(df["col_0"], format="%d/%m/%Y", errors="coerce")

    # garder uniquement les lignes qui ont une date
    df = df[df["date"].notna()].copy()

    # col_1 = statut (si présent)
    if "col_1" in df.columns:
        df = df.rename(columns={"col_1": "statut"})
    else:
        df["statut"] = None

    # supprimer colonnes totalement vides
    df = df.dropna(axis=1, how="all")

    # colonnes de pas de temps = toutes sauf date/statut/col_0
    time_cols = [c for c in df.columns if c not in ["date", "statut", "col_0"]]
    if not time_cols:
        raise ValueError(f"[{year}] Aucune colonne de pas de temps détectée.")

    # melt en long
    df_long = df.melt(
        id_vars=["date", "statut"],
        value_vars=time_cols,
        var_name="slot_col",
        value_name="load_mw",
    )

    # ordre stable
    df_long = df_long.sort_values(["date", "slot_col"]).reset_index(drop=True)

    # index du pas de temps dans la journée
    df_long["slot_index"] = df_long.groupby("date").cumcount()

    # datetime à partir de la date + slot_index * step_minutes
    df_long["datetime"] = df_long["date"] + pd.to_timedelta(
        df_long["slot_index"] * step_minutes, unit="m"
    )

    df_long["year"] = year
    df_long = df_long[["datetime", "date", "year", "statut", "slot_index", "load_mw"]]

    return df_long


def process_year(raw_cons_dir: Path, processed_dir: Path, year: int, step_minutes: int = 30) -> None:
    df_raw = load_year_raw(raw_cons_dir, year)
    if df_raw is None:
        print(f"[{year}] Aucun DataFrame brut disponible, on passe.")
        return

    df_long = tidy_year(df_raw, year, step_minutes=step_minutes)

    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / f"consommation_{year}_long.csv"
    df_long.to_csv(out_path, index=False)

    print(f"[{year}] Sauvegardé : {out_path} ({df_long.shape[0]} lignes)")


def parse_years(years: str) -> list[int]:
    """
    Accepte "2010-2024" ou "2020,2021,2024".
    """
    years = years.strip()
    if "-" in years:
        a, b = years.split("-", 1)
        start = int(a)
        end = int(b)
        if end < start:
            raise ValueError("Plage d'années invalide.")
        return list(range(start, end + 1))
    return [int(y.strip()) for y in years.split(",") if y.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Lire (zip/xls) et convertir RTE consommation en format long.")
    p.add_argument("--base-dir", default=".", help="Racine du projet (défaut: .)")
    p.add_argument("--raw-cons-dir", default="data/raw/consommation", help="Dossier des fichiers bruts consommation")
    p.add_argument("--processed-dir", default="data/processed", help="Dossier de sortie")
    p.add_argument("--years", default="2010-2024", help='Ex: "2010-2024" ou "2020,2021,2024"')
    p.add_argument("--step-minutes", type=int, default=30, help="Pas de temps en minutes (défaut: 30)")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_cons_dir = (base_dir / args.raw_cons_dir).resolve()
    processed_dir = (base_dir / args.processed_dir).resolve()

    years = parse_years(args.years)

    print(f"Base dir       : {base_dir}")
    print(f"Raw cons dir   : {raw_cons_dir}")
    print(f"Processed dir  : {processed_dir}")
    print(f"Years          : {years}")
    print(f"Step (minutes) : {args.step_minutes}")

    for y in years:
        process_year(raw_cons_dir, processed_dir, y, step_minutes=args.step_minutes)


if __name__ == "__main__":
    main()
