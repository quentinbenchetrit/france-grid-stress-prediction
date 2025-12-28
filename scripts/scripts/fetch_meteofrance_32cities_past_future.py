#!/usr/bin/env python3
"""
Fetch Open-Meteo "Meteo-France" model data for the same 32 cities:
- past_days: last N days (archived forecasts)
- forecast_days: next N days (forecast)

Endpoint:
https://api.open-meteo.com/v1/meteofrance

Outputs:
CSV with columns:
date, temperature_2m, wind_speed_10m, direct_radiation, diffuse_radiation, cloud_cover, city
"""

from __future__ import annotations

import argparse
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


# --- CONFIG: SAME 32 CITIES (name, lat, lon) ---
CITIES_CONFIG = [
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Lyon", "lat": 45.7640, "lon": 4.8357},
    {"name": "Marseille - Aix-en-Provence", "lat": 43.2965, "lon": 5.3698},
    {"name": "Toulouse", "lat": 43.6047, "lon": 1.4442},
    {"name": "Bordeaux", "lat": 44.8378, "lon": -0.5792},
    {"name": "Lille (partie française)", "lat": 50.6292, "lon": 3.0573},
    {"name": "Nice", "lat": 43.7102, "lon": 7.2620},
    {"name": "Nantes", "lat": 47.2184, "lon": -1.5536},
    {"name": "Strasbourg (partie française)", "lat": 48.5734, "lon": 7.7521},
    {"name": "Rennes", "lat": 48.1173, "lon": -1.6778},
    {"name": "Grenoble", "lat": 45.1885, "lon": 5.7245},
    {"name": "Rouen", "lat": 49.4431, "lon": 1.0993},
    {"name": "Toulon", "lat": 43.1242, "lon": 5.9280},
    {"name": "Montpellier", "lat": 43.6119, "lon": 3.8772},
    {"name": "Douai - Lens", "lat": 50.3700, "lon": 3.0800},
    {"name": "Avignon", "lat": 43.9493, "lon": 4.8055},
    {"name": "Saint-Étienne", "lat": 45.4397, "lon": 4.3872},
    {"name": "Tours", "lat": 47.3941, "lon": 0.6848},
    {"name": "Clermont-Ferrand", "lat": 45.7772, "lon": 3.0870},
    {"name": "Orléans", "lat": 47.9025, "lon": 1.9090},
    {"name": "Nancy", "lat": 48.6921, "lon": 6.1844},
    {"name": "Angers", "lat": 47.4784, "lon": -0.5632},
    {"name": "Caen", "lat": 49.1829, "lon": -0.3707},
    {"name": "Metz", "lat": 49.1193, "lon": 6.1757},
    {"name": "Dijon", "lat": 47.3220, "lon": 5.0415},
    {"name": "Valenciennes (partie française)", "lat": 50.3590, "lon": 3.5230},
    {"name": "Béthune", "lat": 50.5330, "lon": 2.6330},
    {"name": "Le Mans", "lat": 48.0061, "lon": 0.1996},
    {"name": "Genève - Annemasse (partie française)", "lat": 46.1944, "lon": 6.2378},
    {"name": "Perpignan", "lat": 42.6887, "lon": 2.8948},
    {"name": "Reims", "lat": 49.2583, "lon": 4.0317},
    {"name": "Brest", "lat": 48.3904, "lon": -4.4861},
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch Open-Meteo Meteo-France model (past_days + forecast_days) for 32 French cities."
    )
    parser.add_argument("--past-days", type=int, default=7, help="Number of past days to include (archived forecasts).")
    parser.add_argument("--forecast-days", type=int, default=7, help="Number of future days to include.")
    parser.add_argument("--timezone", default="UTC", help="Timezone for timestamps (e.g. UTC, Europe/Paris).")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default auto-named in current dir).",
    )
    args = parser.parse_args()

    out = args.out or f"weather_32_cities_meteofrance_{args.past_days}past_{args.forecast_days}future.csv"

    # Cache + retries
    cache_session = requests_cache.CachedSession(".cache_openmeteo_meteofrance", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/meteofrance"

    lats = [c["lat"] for c in CITIES_CONFIG]
    lons = [c["lon"] for c in CITIES_CONFIG]

    hourly_vars = [
        "temperature_2m",
        "wind_speed_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    ]

    params = {
        "latitude": lats,
        "longitude": lons,
        "hourly": hourly_vars,
        "timezone": args.timezone,
        "past_days": args.past_days,
        "forecast_days": args.forecast_days,
    }

    print(
        f"Fetching Meteo-France model for {len(CITIES_CONFIG)} cities | "
        f"past_days={args.past_days} | forecast_days={args.forecast_days} | tz={args.timezone}"
    )

    try:
        responses = openmeteo.weather_api(url, params=params)
    except Exception as e:
        print(f"Error calling Open-Meteo: {e}")
        return 2

    frames = []

    for i, response in enumerate(responses):
        city_name = CITIES_CONFIG[i]["name"]

        hourly = response.Hourly()

        # Order matches hourly_vars
        temp = hourly.Variables(0).ValuesAsNumpy()
        wind = hourly.Variables(1).ValuesAsNumpy()
        direct_rad = hourly.Variables(2).ValuesAsNumpy()
        diffuse_rad = hourly.Variables(3).ValuesAsNumpy()
        cloud = hourly.Variables(4).ValuesAsNumpy()

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )

        df_city = pd.DataFrame(
            {
                "date": times,
                "temperature_2m": temp,
                "wind_speed_10m": wind,
                "direct_radiation": direct_rad,
                "diffuse_radiation": diffuse_rad,
                "cloud_cover": cloud,
                "city": city_name,
            }
        )

        frames.append(df_city)

    full_df = pd.concat(frames, ignore_index=True)
    full_df.to_csv(out, index=False)

    print(f"Saved: {out}")
    print(f"Cities: {full_df['city'].nunique()} | Rows: {len(full_df):,} | Columns: {len(full_df.columns)}")
    print(f"Time span: {full_df['date'].min()} -> {full_df['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
