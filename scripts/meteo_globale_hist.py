import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# --- CONFIGURATION ---
TARGET_YEAR = 1997  # Changez ici (ex: 2023) pour récupérer une autre année complète
START_DATE = f"{TARGET_YEAR}-01-01"
END_DATE = f"{TARGET_YEAR}-12-31"
OUTPUT_FILE = f"weather_32_cities_historical_{TARGET_YEAR}.csv"

# URL de l'API Historical Weather (telle que documentée dans votre lien)
URL = "https://archive-api.open-meteo.com/v1/archive"

# Liste EXACTE des 32 agglomérations
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
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Extraction des coordonnées pour la requête groupée
lats = [city["lat"] for city in CITIES_CONFIG]
lons = [city["lon"] for city in CITIES_CONFIG]

params = {
    "latitude": lats,
    "longitude": lons,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "hourly": [
        "temperature_2m", 
        "wind_speed_10m", 
        "direct_radiation", 
        "diffuse_radiation", 
        "cloud_cover"
    ],
    "timezone": "UTC"
}

print(f"🌍 Lancement de la récupération Historical Weather ({START_DATE} au {END_DATE})")
print(f"   Cibles : {len(CITIES_CONFIG)} agglomérations")

try:
    # Appel à l'endpoint 'archive' (Historical Weather API)
    responses = openmeteo.weather_api(URL, params=params)
except Exception as e:
    print(f"❌ Erreur lors de l'appel API : {e}")
    exit()

# --- TRAITEMENT ET FUSION ---
final_df_list = []

for i, response in enumerate(responses):
    city_name = CITIES_CONFIG[i]["name"]
    print(f"   Traitement : {city_name}...")

    hourly = response.Hourly()
    
    # Données brutes (optimisé via Numpy)
    hourly_data = {
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
        "city": city_name
    }
    
    final_df_list.append(pd.DataFrame(data=hourly_data))

# --- SAUVEGARDE ---
print("💾 Fusion et sauvegarde...")
full_df = pd.concat(final_df_list, ignore_index=True)
full_df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Fichier généré : {OUTPUT_FILE}")
print(f"   Dimensions : {full_df.shape}")