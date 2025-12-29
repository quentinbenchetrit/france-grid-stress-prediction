#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

# -------------------------
# Timezone policy (UTC-only output)
# -------------------------
LOCAL_TZ = "Europe/Paris"
UTC_TZ = "UTC"

# Regex souple pour la date
DATE_BLOCK_RE = re.compile(
    r"(?:r.al|donn).*?(\d{2}/\d{2}/\d{4})",
    re.IGNORECASE,
)

# Regex horaire (00:00-01:00)
HOUR_INTERVAL_RE = re.compile(r"^\s*(\d{1,2}:\d{2})\s*[-–]\s*(\d{1,2}:\d{2})\s*$")


def parse_years(years: str) -> list[int]:
    years = years.strip()
    if "-" in years:
        a, b = years.split("-", 1)
        start = int(a)
        end = int(b)
        return list(range(start, end + 1))
    return [int(y.strip()) for y in years.split(",") if y.strip()]


def make_unique(header: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in header:
        h = str(h).strip()
        if h == "" or h.lower() == "nan":
            h = "Unnamed"
        if h not in seen:
            seen[h] = 1
            out.append(h)
        else:
            seen[h] += 1
            out.append(f"{h}__{seen[h]}")
    return out


def read_rte_realisation_file(path: Path) -> pd.DataFrame:
    """
    Lit le fichier en mode 'brut' ligne par ligne pour éviter que Pandas
    ne saute la ligne d'en-tête à cause du changement de nombre de colonnes.
    """
    b = path.read_bytes()

    # 1. Essai lecture Excel natif (pour les vrais .xlsx récents)
    if b.startswith(b"PK"):
        return pd.read_excel(path, header=None, engine="openpyxl")
    if b.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return pd.read_excel(path, header=None, engine="xlrd")

    # 2. Décodage du texte (Gestion des encodages pourris de RTE)
    text = None
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            text = b.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if text is None:
        raise ValueError("Impossible de décoder le fichier (ni utf-8, ni cp1252, ni latin-1)")

    # 3. Parsing Manuel (robuste pour TSV/CSV mal formés)
    lines = text.splitlines()
    lines = [l for l in lines if l.strip()]
    if not lines:
        return pd.DataFrame()

    sample_line = lines[min(100, len(lines) - 1)]
    sep = "\t"
    if sample_line.count(";") > sample_line.count("\t"):
        sep = ";"
    elif sample_line.count(",") > sample_line.count("\t"):
        sep = ","

    data = []
    max_cols = 0
    for line in lines:
        cells = line.split(sep)
        cells = [c.strip().strip('"') for c in cells]
        max_cols = max(max_cols, len(cells))
        data.append(cells)

    aligned_data = []
    for row in data:
        if len(row) < max_cols:
            row += [""] * (max_cols - len(row))
        aligned_data.append(row)

    return pd.DataFrame(aligned_data)


def local_naive_to_utc(dt_series: pd.Series) -> pd.Series:
    """
    Convert naive local timestamps (implicit Europe/Paris) to timezone-aware UTC.

    DST handling:
    - ambiguous="NaT": repeated local times (fall-back) are marked as NaT and dropped later
    - nonexistent="shift_forward": missing local times (spring-forward) are shifted forward
    """
    s = pd.to_datetime(dt_series, errors="coerce")
    s = s.dt.tz_localize(LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward")
    return s.dt.tz_convert(UTC_TZ)


def tidy_year(df_raw: pd.DataFrame, year: int) -> pd.DataFrame:
    df = df_raw.copy()

    # Renommage générique
    df.columns = [f"col_{i}" for i in range(df.shape[1])]
    idx = list(df.index)
    n = len(idx)

    # Helpers
    def get_row_vals(r):
        return [str(x).strip() for x in df.loc[r].tolist()]

    def row_text_joined(r):
        return " ".join(get_row_vals(r)).lower()

    rows_out: list[dict] = []
    i = 0

    while i < n:
        # 1. Chercher la DATE
        txt = row_text_joined(idx[i])
        m = DATE_BLOCK_RE.search(txt)

        if not m:
            i += 1
            continue

        day_str = m.group(1)
        day = pd.to_datetime(day_str, format="%d/%m/%Y", errors="coerce")
        if pd.isna(day):
            i += 1
            continue

        # 2. Chercher le HEADER (Heures, Biomasse, etc.)
        j = i + 1
        found_header = False
        header_idx = -1

        while j < n and j < i + 10:
            row_str = row_text_joined(idx[j])
            if "heures" in row_str or "biomasse" in row_str or "filière" in row_str or "nucléaire" in row_str:
                found_header = True
                header_idx = j
                break
            j += 1

        if not found_header:
            print(f"[{year}] Avertissement: Header non trouvé pour la date {day_str}. Saut du bloc.")
            i += 1
            continue

        header_raw = get_row_vals(idx[header_idx])
        header = make_unique(header_raw)

        # Trouver la colonne "Heures"
        heures_pos = -1
        for pos, val in enumerate(header_raw):
            if "heures" in val.lower():
                heures_pos = pos
                break
        if heures_pos == -1:
            heures_pos = 0  # Fallback

        # 3. Parser les DONNÉES
        k = header_idx + 1
        while k < n:
            if DATE_BLOCK_RE.search(row_text_joined(idx[k])):
                break

            cells = get_row_vals(idx[k])
            if not "".join(cells).strip():
                k += 1
                continue

            hour_candidate = cells[heures_pos] if heures_pos < len(cells) else ""
            hm = HOUR_INTERVAL_RE.match(hour_candidate)

            if not hm:
                for c in cells:
                    hm = HOUR_INTERVAL_RE.match(c)
                    if hm:
                        hour_candidate = c
                        break

            if not hm:
                k += 1
                continue

            start_hhmm = hm.group(1)

            # Build naive local timestamp (Europe/Paris implied)
            full_dt = f"{day.strftime('%Y-%m-%d')} {start_hhmm}"
            dt_local_naive = pd.to_datetime(full_dt, format="%Y-%m-%d %H:%M", errors="coerce")

            # If parsing failed, skip row
            if pd.isna(dt_local_naive):
                k += 1
                continue

            # Extraction des valeurs
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))

            for col_idx, (tech, val) in enumerate(zip(header, cells)):
                tech_lower = tech.lower()
                if "heures" in tech_lower or "unnamed" in tech_lower or col_idx == heures_pos:
                    continue

                val_clean = val.replace(",", ".").replace("\xa0", "").replace(" ", "")
                if val_clean in ["-", ""]:
                    val_clean = "0"

                v = pd.to_numeric(val_clean, errors="coerce")

                rows_out.append(
                    {
                        "datetime": dt_local_naive,          # will be converted to UTC after extraction
                        "date": pd.to_datetime(day.date()),  # will be converted to UTC midnight after extraction
                        "year": year,
                        "hour_interval": hour_candidate,
                        "technology": tech,
                        "value_mw": v,
                    }
                )

            k += 1

        i = k

    if not rows_out:
        raise ValueError(f"[{year}] 0 lignes extraites. Vérifiez le format.")

    df_out = pd.DataFrame(rows_out)

    # ---- UTC NORMALIZATION (CORE FIX) ----
    df_out["datetime"] = local_naive_to_utc(df_out["datetime"])
    df_out["date"] = local_naive_to_utc(df_out["date"])

    # Drop ambiguous timestamps (DST fall-back) explicitly
    df_out = df_out[df_out["datetime"].notna()].copy()

    # Sort + deduplicate (if any)
    df_out = df_out.sort_values(["datetime", "technology"]).reset_index(drop=True)
    if df_out["datetime"].duplicated().any():
        df_out = (
            df_out.groupby(["datetime", "date", "year", "hour_interval", "technology"], as_index=False)["value_mw"]
            .mean()
        )

    return df_out


def process_year(raw_prod_dir: Path, out_dir: Path, year: int) -> None:
    path = None
    for ext in [".xls", ".xlsx"]:
        p = raw_prod_dir / f"RealisationDonneesProduction_{year}{ext}"
        if p.exists():
            path = p
            break

    if path is None:
        print(f"[{year}] Fichier introuvable.")
        return

    print(f"[{year}] Lecture de {path.name}...")
    try:
        df_raw = read_rte_realisation_file(path)
        df_long = tidy_year(df_raw, year)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Output in interim/production, UTC-only, with _utc before .csv
        out_path = out_dir / f"production_realisation_{year}_long_utc.csv"
        df_long.to_csv(out_path, index=False)

        print(f"[{year}] SUCCÈS : {len(df_long):,} lignes sauvegardées -> {out_path}")
        print(f"[{year}] Coverage UTC: {df_long['datetime'].min()} -> {df_long['datetime'].max()}")
        print(f"[{year}] Duplicate datetimes: {int(df_long['datetime'].duplicated().sum())}")

    except Exception as e:
        print(f"[{year}] ERREUR : {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="/home/onyxia/work/france-grid-stress-prediction")
    p.add_argument("--raw-prod-dir", default="data/raw/production")
    # Force output to interim/production as requested
    p.add_argument("--out-dir", default="data/interim/production")
    p.add_argument("--years", default="2015-2024")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_prod_dir = (base_dir / args.raw_prod_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    years = parse_years(args.years)

    print(f"Dossier source : {raw_prod_dir}")
    print(f"Dossier sortie : {out_dir}")
    print(f"Années : {years}")
    print(f"Timezone policy: interpret naive datetimes as {LOCAL_TZ} -> convert to {UTC_TZ}")
    print("DST policy: ambiguous local times -> dropped (NaT), nonexistent local times -> shifted forward")

    for y in years:
        process_year(raw_prod_dir, out_dir, y)


if __name__ == "__main__":
    main()
