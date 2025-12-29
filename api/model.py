from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import joblib

@dataclass
class ModelBundle:
    model: object
    feature_names: list[str] | None = None

def load_model(model_path: str | Path, feature_names: list[str] | None = None) -> ModelBundle:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    return ModelBundle(model=model, feature_names=feature_names)
