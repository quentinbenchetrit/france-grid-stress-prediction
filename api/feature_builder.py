from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import pandas as pd

@dataclass
class WeatherInput:
    timestamp: datetime
    temp_c: float
    wind_speed_ms: float
    humidity: float | None = None
    pressure_hpa: float | None = None

def build_features(w: WeatherInput) -> pd.DataFrame:
    # Exemples de features (à aligner avec ton training !)
    ts = w.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    row = {
        "temp_c": w.temp_c,
        "wind_speed_ms": w.wind_speed_ms,
        "humidity": w.humidity if w.humidity is not None else 0.0,
        "pressure_hpa": w.pressure_hpa if w.pressure_hpa is not None else 0.0,
        "hour": ts.hour,
        "dayofweek": ts.weekday(),  # 0=lundi
        "month": ts.month,
        "is_weekend": 1 if ts.weekday() >= 5 else 0,
    }

    return pd.DataFrame([row])
