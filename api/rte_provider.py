from __future__ import annotations
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Chemins
HISTORY_FILE = Path("data/rte_history.csv")
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# RTE Config
TOKEN_URL = "https://digital.iservices.rte-france.com/token/oauth/"
API_URL = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"

def get_oauth_token():
    client_id = os.getenv("RTE_CLIENT_ID")
    client_secret = os.getenv("RTE_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        raise ValueError("RTE_CLIENT_ID et RTE_CLIENT_SECRET doivent être définis en variables d'environnement.")

    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=10
    )
    resp.raise_for_status()
    return resp.json()["access_token"]

def update_history_data():
    """
    Télécharge les 8 derniers jours de données RTE et met à jour le fichier local.
    Nécessaire pour calculer les lags (J-1, J-7).
    """
    token = get_oauth_token()
    
    # On demande 8 jours pour avoir assez de marge pour le lag 168h (7 jours)
    end_dt = datetime.now().astimezone()
    start_dt = end_dt - timedelta(days=8)

    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "start_date": start_dt.isoformat(timespec="seconds"),
        "end_date": end_dt.isoformat(timespec="seconds"),
        "type": "REALISED"
    }

    try:
        resp = requests.get(API_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Warning: Impossible de fetch RTE ({e}). On utilise le cache local.")
        return

    records = []
    for block in data.get("short_term", []):
        for v in block.get("values", []):
            records.append({
                "date": v["start_date"], # String ISO
                "val": v["value"]
            })
    
    if not records:
        return

    new_df = pd.DataFrame(records)
    new_df["date"] = pd.to_datetime(new_df["date"], utc=True)
    new_df = new_df.set_index("date").sort_index()

    # Si un fichier existe, on merge, sinon on crée
    if HISTORY_FILE.exists():
        old_df = pd.read_csv(HISTORY_FILE, index_col=0, parse_dates=True)
        # Combine old and new, keep last on duplicate index
        combined = new_df.combine_first(old_df)
        # On ne garde que les 10 derniers jours pour éviter que le fichier grossisse à l'infini
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
        combined = combined[combined.index > cutoff]
        combined.to_csv(HISTORY_FILE)
    else:
        new_df.to_csv(HISTORY_FILE)

def get_consumption_features() -> dict:
    """
    Lit l'historique local et calcule les features complexes (lags, rolling).
    Retourne les valeurs pour l'instant 'T' (le plus récent possible).
    """
    if not HISTORY_FILE.exists():
        # Fallback critique si 0 données (Cold start)
        return {k: 0.0 for k in [
            "load_lag_1h", "load_lag_24h", "load_lag_48h", "load_lag_168h",
            "load_roll_mean_24h", "load_roll_std_24h", 
            "load_roll_mean_168h", "load_roll_std_168h"
        ]}

    df = pd.read_csv(HISTORY_FILE, index_col=0, parse_dates=True)
    df = df.sort_index()
    
    # Calcul des features sur toute la série temporelle
    # (Pandas gère très bien ça vectoriellement)
    df["load_lag_1h"] = df["val"].shift(2 * 1)    # 30min step * 2 = 1h
    df["load_lag_24h"] = df["val"].shift(2 * 24)
    df["load_lag_48h"] = df["val"].shift(2 * 48)
    df["load_lag_168h"] = df["val"].shift(2 * 168) # 7 jours

    # Rolling windows (sur 24h = 48 points de 30min)
    df["load_roll_mean_24h"] = df["val"].rolling(window=48).mean()
    df["load_roll_std_24h"]  = df["val"].rolling(window=48).std()
    
    # Rolling windows (sur 168h = 336 points)
    df["load_roll_mean_168h"] = df["val"].rolling(window=336).mean()
    df["load_roll_std_168h"]  = df["val"].rolling(window=336).std()

    # On prend la dernière ligne (le point le plus récent connu)
    last_row = df.iloc[-1]

    # Construction du dict de retour
    return {
        "load_lag_1h": float(last_row["load_lag_1h"]),
        "load_lag_24h": float(last_row["load_lag_24h"]),
        "load_lag_48h": float(last_row["load_lag_48h"]),
        "load_lag_168h": float(last_row["load_lag_168h"]),
        
        "load_roll_mean_24h": float(last_row["load_roll_mean_24h"]),
        "load_roll_std_24h": float(last_row["load_roll_std_24h"]),
        "load_roll_mean_168h": float(last_row["load_roll_mean_168h"]),
        "load_roll_std_168h": float(last_row["load_roll_std_168h"]),
    }