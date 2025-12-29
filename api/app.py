from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import load_model

# --- Paths (projet) ---
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "xgb_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "metadata.json"

# --- FastAPI app ---
# root_path nécessaire derrière /proxy/8000
app = FastAPI(
    title="Grid Stress Prediction API",
    version="1.0.0",
    root_path="/proxy/8000",
)

# --- Load model ---
bundle = load_model(MODEL_PATH)

# --- Load metadata (features) ---
if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"metadata.json introuvable: {METADATA_PATH}\n"
        "Crée-le dans models/ avec au minimum la clé 'features' (liste des colonnes du training)."
    )

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    METADATA = json.load(f)

if "features" not in METADATA or not isinstance(METADATA["features"], list) or len(METADATA["features"]) == 0:
    raise ValueError(
        "metadata.json doit contenir une clé 'features' non vide, ex:\n"
        '{ "features": ["temperature_2m", "..."], "target": "..." }'
    )

FEATURES: List[str] = METADATA["features"]
TARGET: Optional[str] = METADATA.get("target")


# --- Schemas ---
class PredictRequest(BaseModel):
    timestamp: Optional[str] = Field(
        default=None,
        description="Optionnel. Timestamp ISO (ex: 2024-01-15T12:00:00Z).",
    )
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionnaire: nom_feature -> valeur. Doit contenir TOUTES les features attendues par le modèle.",
    )


class PredictResponse(BaseModel):
    timestamp: Optional[str]
    prediction: float
    model: str = "xgboost"
    target: Optional[str] = None
    missing_features: List[str] = []
    extra_features: List[str] = []


# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "API running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "target": TARGET,
        "n_features": len(FEATURES),
    }


@app.get("/model/info")
def model_info():
    # pratique pour vérifier FEATURES côté Swagger
    return {
        "target": TARGET,
        "n_features": len(FEATURES),
        "features": FEATURES,
        **{k: v for k, v in METADATA.items() if k not in ["features"]},
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    provided = req.features

    missing = [c for c in FEATURES if c not in provided]
    extra = [c for c in provided.keys() if c not in FEATURES]

    if missing:
        # On refuse proprement si features manquantes
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Missing required features for this model",
                "missing_features": missing,
                "extra_features": extra,
                "expected_feature_count": len(FEATURES),
                "provided_feature_count": len(provided),
            },
        )

    # DataFrame dans l'ordre exact du training
    X = pd.DataFrame([{c: provided[c] for c in FEATURES}], columns=FEATURES)


    try:
        yhat = float(bundle.model.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictResponse(
        timestamp=req.timestamp,
        prediction=yhat,
        target=TARGET,
        missing_features=[],
        extra_features=extra,
    )


# Ajoute l'import en haut de app.py
from .feature_builder import build_live_features

# ... (le reste du code) ...

# Remplace l'ancien endpoint realtime par celui-ci :
@app.get("/predict/realtime", response_model=PredictResponse)
def predict_realtime():
    """
    Prédiction en direct :
    1. Récupère la météo live (Open-Meteo)
    2. Récupère l'historique conso (RTE API)
    3. Calcule toutes les features et prédit.
    """
    try:
        # 1. Construction automatique des features
        X = build_live_features()
        
        # 2. Vérification de l'ordre des colonnes (Sécurité)
        # On s'assure que X a bien les colonnes dans l'ordre de FEATURES
        # (FEATURES vient de ton metadata.json chargé plus haut)
        X = X[FEATURES] 
        
        # 3. Prédiction
        yhat = float(bundle.model.predict(X)[0])
        
        return PredictResponse(
            timestamp=datetime.now().isoformat(),
            prediction=yhat,
            target=TARGET,
            missing_features=[],
            extra_features=[],
            model="xgboost-realtime"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realtime prediction failed: {str(e)}")

