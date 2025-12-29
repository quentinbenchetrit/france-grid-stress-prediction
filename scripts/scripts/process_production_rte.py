#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd

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

    # 3. Parsing Manuel (Le plus robuste pour les TSV mal formés)
    # On découpe ligne par ligne, puis par tabulation.
    lines = text.splitlines()
    
    # On nettoie les lignes vides
    lines = [l for l in lines if l.strip()]
    
    if not lines:
        return pd.DataFrame()

    # Détection du séparateur : on regarde ce qui sépare le mieux sur une ligne de données
    # On prend une ligne au milieu du fichier pour tester
    sample_line = lines[min(100, len(lines)-1)]
    sep = "\t" # Par défaut : tabulation
    if sample_line.count(";") > sample_line.count("\t"):
        sep = ";"
    elif sample_line.count(",") > sample_line.count("\t"):
        sep = ","
        
    # Découpage
    data = []
    max_cols = 0
    for line in lines:
        # On découpe
        cells = line.split(sep)
        # On nettoie les espaces/guillemets
        cells = [c.strip().strip('"') for c in cells]
        if len(cells) > max_cols:
            max_cols = len(cells)
        data.append(cells)

    # Alignement : on s'assure que toutes les lignes ont le même nombre de colonnes
    # (nécessaire pour créer le DataFrame)
    aligned_data = []
    for row in data:
        if len(row) < max_cols:
            row += [""] * (max_cols - len(row))
        aligned_data.append(row)

    df = pd.DataFrame(aligned_data)
    return df


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
        # On cherche dans les 10 lignes suivantes
        j = i + 1
        found_header = False
        header_idx = -1
        
        while j < n and j < i + 10:
            row_str = row_text_joined(idx[j])
            # Mots clés typiques d'un header RTE
            if "heures" in row_str or "biomasse" in row_str or "filière" in row_str or "nucléaire" in row_str:
                found_header = True
                header_idx = j
                break
            j += 1
        
        if not found_header:
            print(f"[{year}] Avertissement: Header non trouvé pour la date {day_str}. Saut du bloc.")
            i += 1
            continue

        # Récupération et nettoyage du header
        header_raw = get_row_vals(idx[header_idx])
        header = make_unique(header_raw)
        
        # Trouver la colonne "Heures"
        heures_pos = -1
        for pos, val in enumerate(header_raw):
            if "heures" in val.lower():
                heures_pos = pos
                break
        if heures_pos == -1: 
            heures_pos = 0 # Fallback

        # 3. Parser les DONNÉES (lignes suivantes)
        k = header_idx + 1
        while k < n:
            # Stop si on trouve une nouvelle date
            if DATE_BLOCK_RE.search(row_text_joined(idx[k])):
                break
            
            cells = get_row_vals(idx[k])
            
            # Ligne vide ?
            if not "".join(cells).strip():
                k += 1
                continue

            # Extraire l'heure
            hour_candidate = cells[heures_pos] if heures_pos < len(cells) else ""
            hm = HOUR_INTERVAL_RE.match(hour_candidate)
            
            # Si pas trouvé dans la colonne attendue, on cherche partout
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
            full_dt = f"{day.strftime('%Y-%m-%d')} {start_hhmm}"
            dt = pd.to_datetime(full_dt, format="%Y-%m-%d %H:%M", errors="coerce")

            # Extraction des valeurs
            if len(cells) < len(header):
                cells += [""] * (len(header) - len(cells))

            for col_idx, (tech, val) in enumerate(zip(header, cells)):
                # Ignorer colonnes inutiles
                tech_lower = tech.lower()
                if "heures" in tech_lower or "unnamed" in tech_lower or col_idx == heures_pos:
                    continue
                
                # Nettoyage valeur (virgule, espaces insécables)
                val_clean = val.replace(",", ".").replace("\xa0", "").replace(" ", "")
                if val_clean in ["-", ""]:
                    val_clean = "0"
                
                v = pd.to_numeric(val_clean, errors="coerce")
                
                rows_out.append({
                    "datetime": dt,
                    "date": day.date(),
                    "year": year,
                    "hour_interval": hour_candidate,
                    "technology": tech,
                    "value_mw": v
                })
            
            k += 1
        
        i = k

    if not rows_out:
        raise ValueError(f"[{year}] 0 lignes extraites. Vérifiez le format.")

    return pd.DataFrame(rows_out)


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
        out_path = out_dir / f"production_realisation_{year}_long.csv"
        df_long.to_csv(out_path, index=False)
        
        print(f"[{year}] SUCCÈS : {len(df_long):,} lignes sauvegardées -> {out_path.name}")
        
    except Exception as e:
        print(f"[{year}] ERREUR : {e}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="/home/onyxia/work/france-grid-stress-prediction")
    p.add_argument("--raw-prod-dir", default="data/raw/production")
    p.add_argument("--out-dir", default="data/interim/production")
    p.add_argument("--years", default="2015-2024")
    args = p.parse_args()

    base_dir = Path(args.base_dir).resolve()
    raw_prod_dir = (base_dir / args.raw_prod_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()
    years = parse_years(args.years)

    print(f"Dossier source : {raw_prod_dir}")
    print(f"Années : {years}")

    for y in years:
        process_year(raw_prod_dir, out_dir, y)


if __name__ == "__main__":
    main()