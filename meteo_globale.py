import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# --- CONFIGURATION ---
TARGET_YEAR = 2014  # Remplacer par 2023 pour obtenir l'année précédente
OUTPUT_FILE = f"weather_32_cities_{TARGET_YEAR}.csv"

# Liste EXACTE des 32 agglomérations fournie (Nom, Latitude, Longitude)
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
    {"name": "Brest", "lat": 48.3904, "lon": -4.4861}
]

# --- PRÉPARATION DES REQUÊTES ---

# 1. Configuration Client API avec Cache
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. Construction des listes de coordonnées pour la requête groupée
url = "https://archive-api.open-meteo.com/v1/archive"
lats = [city["lat"] for city in CITIES_CONFIG]
lons = [city["lon"] for city in CITIES_CONFIG]

params = {
    "latitude": lats,
    "longitude": lons,
    "start_date": f"{TARGET_YEAR}-01-01",
    "end_date": f"{TARGET_YEAR}-12-31",
    "hourly": [
        "temperature_2m", 
        "wind_speed_10m", 
        "direct_radiation", 
        "diffuse_radiation", 
        "cloud_cover"
    ],
    "timezone": "UTC"
}

print(f"🌍 Lancement de la récupération pour les {len(CITIES_CONFIG)} agglomérations en {TARGET_YEAR}...")

# 3. Appel API (Une seule requête optimisée)
try:
    responses = openmeteo.weather_api(url, params=params)
except Exception as e:
    print(f"❌ Erreur lors de l'appel API : {e}")
    exit()

# --- TRAITEMENT ET FUSION ---

final_df_list = []

# On parcourt les réponses dans l'ordre (qui correspond à l'ordre de CITIES_CONFIG)
for i, response in enumerate(responses):
    city_name = CITIES_CONFIG[i]["name"]
    print(f"   Traitement : {city_name}...")

    # Récupération des données horaires
    hourly = response.Hourly()
    
    # Extraction optimisée via Numpy
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(2).ValuesAsNumpy()
    hourly_diffuse_radiation = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(4).ValuesAsNumpy()

    # Création de l'index temporel
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    }
    
    # Remplissage des colonnes
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["direct_radiation"] = hourly_direct_radiation
    hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    
    # Création du DataFrame pour cette ville
    df_city = pd.DataFrame(data=hourly_data)
    df_city["city"] = city_name # Ajout du nom de la ville pour identification
    
    final_df_list.append(df_city)

# 4. Fusion finale et Export
print("💾 Fusion et sauvegarde des données...")
full_df = pd.concat(final_df_list, ignore_index=True)

# Sauvegarde CSV
full_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Terminé ! Fichier généré : {OUTPUT_FILE}")
print(f"   Villes traitées : {len(full_df['city'].unique())}")
print(f"   Dimensions totales : {full_df.shape}")