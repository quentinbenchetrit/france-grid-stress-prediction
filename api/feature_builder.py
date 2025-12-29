from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# On importe tes deux nouveaux providers

from .rte_provider import update_history_data, get_consumption_features
from .weather_provider import get_live_weather_features

def build_live_features() -> pd.DataFrame:
    """
    Construit le vecteur de features complet pour l'instant T (maintenant).
    Orchestre la récupération Météo + RTE + Calculs mathématiques.
    """
    
    # 1. Mise à jour des données RTE (télécharge les dernières 24h si besoin)
   
    try:
        update_history_data()
    except Exception as e:
        print(f"⚠️ Warning: Impossible de mettre à jour RTE ({e}). On utilise le cache existant.")

    # 2. Récupération des inputs bruts
    # a) Features de consommation (Lags & Rolling)
    cons_features = get_consumption_features()
    
    # b) Features météo (Moyenne nationale)
    weather_features = get_live_weather_features()
    
    # 3. Gestion du Temps (Date actuelle UTC)
    # On utilise l'heure de la météo pour être cohérent, ou 'now'
    ts = weather_features.get("timestamp_weather")
    if ts is None:
        ts = datetime.now(timezone.utc)
    
    # Extraction des composants temporels
    hour = ts.hour
    dayofweek = ts.weekday()  # 0=Lundi, 6=Dimanche
    month = ts.month
    dayofyear = ts.timetuple().tm_yday
    
    # 4. Calculs des features cycliques (Maths)
    # C'est crucial pour que le modèle comprenne que 23h est proche de 00h
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    
    doy_sin = np.sin(2 * np.pi * dayofyear / 366.0)
    doy_cos = np.cos(2 * np.pi * dayofyear / 366.0)

    # 5. Assemblage du dictionnaire final
  
    row = {
        # --- Météo ---
        "temperature_2m": weather_features["temperature_2m"],
        "wind_speed_10m": weather_features["wind_speed_10m"],
        "direct_radiation": weather_features["direct_radiation"],
        "diffuse_radiation": weather_features["diffuse_radiation"],
        "cloud_cover": weather_features["cloud_cover"],

        # --- Temps ---
        "hour": hour,
        "dayofweek": dayofweek,
        "is_weekend": 1 if dayofweek >= 5 else 0,
        "month": month,
        "dayofyear": dayofyear,

        # --- Cyclique ---
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,

        # --- Historique Charge (RTE) ---
        "load_lag_1h": cons_features["load_lag_1h"],
        "load_lag_24h": cons_features["load_lag_24h"],
        "load_lag_48h": cons_features["load_lag_48h"],
        "load_lag_168h": cons_features["load_lag_168h"],

        "load_roll_mean_24h": cons_features["load_roll_mean_24h"],
        "load_roll_std_24h": cons_features["load_roll_std_24h"],
        "load_roll_mean_168h": cons_features["load_roll_mean_168h"],
        "load_roll_std_168h": cons_features["load_roll_std_168h"],
    }

    # Création du DataFrame (1 seule ligne)
    df = pd.DataFrame([row])
    
    return df