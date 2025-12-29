from __future__ import annotations
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from datetime import datetime, timedelta, timezone

# Configuration des 32 villes (identique à ton script original)
CITIES_CONFIG = [
    {"lat": 48.8566, "lon": 2.3522},   # Paris
    {"lat": 45.7640, "lon": 4.8357},   # Lyon
    {"lat": 43.2965, "lon": 5.3698},   # Marseille
    {"lat": 43.6047, "lon": 1.4442},   # Toulouse
    {"lat": 44.8378, "lon": -0.5792},  # Bordeaux
    {"lat": 50.6292, "lon": 3.0573},   # Lille
    {"lat": 43.7102, "lon": 7.2620},   # Nice
    {"lat": 47.2184, "lon": -1.5536},  # Nantes
    {"lat": 48.5734, "lon": 7.7521},   # Strasbourg
    {"lat": 48.1173, "lon": -1.6778},  # Rennes
    {"lat": 45.1885, "lon": 5.7245},   # Grenoble
    {"lat": 49.4431, "lon": 1.0993},   # Rouen
    {"lat": 43.1242, "lon": 5.9280},   # Toulon
    {"lat": 43.6119, "lon": 3.8772},   # Montpellier
    {"lat": 50.3700, "lon": 3.0800},   # Douai
    {"lat": 43.9493, "lon": 4.8055},   # Avignon
    {"lat": 45.4397, "lon": 4.3872},   # St-Etienne
    {"lat": 47.3941, "lon": 0.6848},   # Tours
    {"lat": 45.7772, "lon": 3.0870},   # Clermont-Ferrand
    {"lat": 47.9025, "lon": 1.9090},   # Orleans
    {"lat": 48.6921, "lon": 6.1844},   # Nancy
    {"lat": 47.4784, "lon": -0.5632},  # Angers
    {"lat": 49.1829, "lon": -0.3707},  # Caen
    {"lat": 49.1193, "lon": 6.1757},   # Metz
    {"lat": 47.3220, "lon": 5.0415},   # Dijon
    {"lat": 50.3590, "lon": 3.5230},   # Valenciennes
    {"lat": 50.5330, "lon": 2.6330},   # Bethune
    {"lat": 48.0061, "lon": 0.1996},   # Le Mans
    {"lat": 46.1944, "lon": 6.2378},   # Geneve
    {"lat": 42.6887, "lon": 2.8948},   # Perpignan
    {"lat": 49.2583, "lon": 4.0317},   # Reims
    {"lat": 48.3904, "lon": -4.4861},  # Brest
]

# Setup client Open-Meteo avec cache
cache_session = requests_cache.CachedSession('.cache_weather', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_live_weather_features() -> dict:
    """
    Récupère la météo actuelle moyennée sur la France pour les features du modèle.
    Retourne un dictionnaire avec : temperature_2m, wind_speed_10m, radiation, etc.
    """
    url = "https://api.open-meteo.com/v1/meteofrance"
    
    # On demande "l'heure actuelle" + 1h de forecast pour être sûr d'avoir le point
    params = {
        "latitude": [c["lat"] for c in CITIES_CONFIG],
        "longitude": [c["lon"] for c in CITIES_CONFIG],
        "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation", "diffuse_radiation", "cloud_cover"],
        "timezone": "UTC",
        "forecast_days": 1
    }

    responses = openmeteo.weather_api(url, params=params)
    
    frames = []
    for resp in responses:
        hourly = resp.Hourly()
        
        # Extraction des données numpy
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed_10m": hourly.Variables(1).ValuesAsNumpy(),
            "direct_radiation": hourly.Variables(2).ValuesAsNumpy(),
            "diffuse_radiation": hourly.Variables(3).ValuesAsNumpy(),
            "cloud_cover": hourly.Variables(4).ValuesAsNumpy(),
        }
        frames.append(pd.DataFrame(data))

    # Concaténation et MOYENNE nationale par timestamp
    full_df = pd.concat(frames)
    avg_df = full_df.groupby("date").mean().reset_index()

    # On prend la ligne la plus proche de l'heure actuelle
    now = datetime.now(timezone.utc)
    # On cherche l'index le plus proche
    closest_idx = (avg_df['date'] - now).abs().idxmin()
    row = avg_df.loc[closest_idx]

    return {
        "temperature_2m": float(row["temperature_2m"]),
        "wind_speed_10m": float(row["wind_speed_10m"]),
        "direct_radiation": float(row["direct_radiation"]),
        "diffuse_radiation": float(row["diffuse_radiation"]),
        "cloud_cover": float(row["cloud_cover"]),
        "timestamp_weather": row["date"] # Juste pour info/debug
    }