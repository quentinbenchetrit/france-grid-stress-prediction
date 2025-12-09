import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# 1. Configuration du client API avec cache et "retry" (robustesse)
# Le cache permet d'éviter de redemander les mêmes données si on relance le script.
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. Définition des paramètres de la requête
# URL de l'API Archive pour l'historique 2024
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 48.8566,  # Coordonnées de Paris
    "longitude": 2.3522,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "hourly": [
        "temperature_2m",      # Variable principale thermosensibilité 
        "wind_speed_10m",      # Pour l'éolien et le refroidissement 
        "cloud_cover",         # Impact éclairage et solaire 
        "direct_radiation",    # Solaire photovoltaïque 
        "diffuse_radiation"    # Solaire photovoltaïque 
    ],
    "timezone": "UTC"          # Impératif pour l'alignement avec RTE 
}

print("📥 Récupération des données météo pour Paris (2024)...")
responses = openmeteo.weather_api(url, params=params)

# Traitement de la réponse (on prend la première location, ici Paris)
response = responses[0]

print(f"Coordonnées utilisées : {response.Latitude()}°N, {response.Longitude()}°E")
print(f"Zone horaire : {response.Timezone()} ({response.TimezoneAbbreviation()})")

# 3. Traitement des données horaires
hourly = response.Hourly()

# Extraction des valeurs brutes dans des tableaux numpy
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(2).ValuesAsNumpy()
hourly_direct_radiation = hourly.Variables(3).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(4).ValuesAsNumpy()

# Création de l'index temporel
hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
}

# 4. Création du DataFrame
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["direct_radiation"] = hourly_direct_radiation
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation

df = pd.DataFrame(data=hourly_data)

# 5. Sauvegarde des données
output_file = "weather_paris_2024.csv"
# On sauvegarde sans l'index numérique, mais avec la colonne date
df.to_csv(output_file, index=False)

print(f"✅ Données sauvegardées avec succès dans '{output_file}'")
print(f"   Dimensions : {df.shape[0]} lignes (heures) x {df.shape[1]} colonnes")
print("\nAperçu des premières lignes :")
print(df.head())