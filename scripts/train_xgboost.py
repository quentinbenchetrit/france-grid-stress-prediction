#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


# -----------------------
# CONFIG
# -----------------------

@dataclass
class Config:
    project_root: str = "/home/onyxia/work/france-grid-stress-prediction"
    consumption_dir: str = "data/processed"

    # météo: 2010-2014 en historical, 2015-2024 en normal (comme tes fichiers)
    start_year: int = 2010
    end_year: int = 2024

    cities_xlsx: str = "agglomerations.xlsx"


    # colonnes météo attendues (issues de tes scripts meteo_globale*.py)
    weather_vars: Tuple[str, ...] = (
        "temperature_2m",
        "wind_speed_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
    )

    # split temporel
    train_end: str = "2021-12-31 23:00:00"
    val_end: str = "2022-12-31 23:00:00"
    test_end: str = "2024-12-31 23:00:00"

    # features lags/rolling (en heures)
    lags: Tuple[int, ...] = (1, 2, 24, 48, 168)          # 1h,2h,1j,2j,1sem
    roll_windows: Tuple[int, ...] = (24, 168)            # 1j,1sem

    # modèle
    random_state: int = 42

    # sortie
    model_dir: str = "models"
    model_file: str = "hgb_load_forecast.joblib"
    dataset_cache_file: str = "data/processed/dataset_model_hourly.parquet"


# -----------------------
# UTILS
# -----------------------

def _safe_read_excel_population(xlsx_path: str) -> pd.DataFrame:
    """
    Lit l'Excel et retourne un DF (city, pop, weight).
    On se base sur les colonnes vues dans ton fichier :
    - "Agglomération"
    - "Population AU 2017"
    """
    df = pd.read_excel(xlsx_path)
    # normalisation minimale
    df = df.rename(columns={
        "Agglomération": "city",
        "Population AU 2017": "population",
    })
    if "city" not in df.columns or "population" not in df.columns:
        raise ValueError(f"Colonnes attendues introuvables dans {xlsx_path}. Colonnes: {list(df.columns)}")

    df = df[["city", "population"]].copy()
    df["city"] = df["city"].astype(str).str.strip()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    df = df[df["population"] > 0]

    total = df["population"].sum()
    df["weight"] = df["population"] / total
    return df


def _list_weather_files(cfg: Config) -> List[str]:
    files = []
    for y in range(cfg.start_year, cfg.end_year + 1):
        if y <= 2014:
            f = os.path.join(cfg.project_root, f"weather_32_cities_historical_{y}.csv")
        else:
            f = os.path.join(cfg.project_root, f"weather_32_cities_{y}.csv")
        files.append(f)
    return files


def _load_weather(cfg: Config, pop_weights: pd.DataFrame) -> pd.DataFrame:
    """
    Charge météo (toutes années), puis agrège 32 villes -> features France pondérées.
    Sortie: index datetime horaire (UTC), colonnes meteo agrégées.
    """
    weather_files = _list_weather_files(cfg)
    missing = [f for f in weather_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Fichiers météo manquants:\n" + "\n".join(missing))

    dfs = []
    for f in weather_files:
        df = pd.read_csv(f)
        # colonnes attendues: date, city, vars...
        if "date" not in df.columns or "city" not in df.columns:
            raise ValueError(f"Format météo inattendu: {f} (colonnes: {list(df.columns)})")

        for v in cfg.weather_vars:
            if v not in df.columns:
                raise ValueError(f"Colonne météo manquante {v} dans {f}")

        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        df = df.dropna(subset=["date"])
        df["city"] = df["city"].astype(str).str.strip()
        dfs.append(df[["date", "city", *cfg.weather_vars]])

    w = pd.concat(dfs, ignore_index=True)

    # merge poids population
    w = w.merge(pop_weights[["city", "weight"]], on="city", how="left")

    # si certaines villes ne matchent pas exactement, on avertit
    unmatched = w["weight"].isna().mean()
    if unmatched > 0:
        # on liste quelques exemples
        ex = w.loc[w["weight"].isna(), "city"].drop_duplicates().head(10).tolist()
        warnings.warn(
            f"Attention: {unmatched:.2%} des lignes météo n'ont pas de poids population (mismatch noms ville). "
            f"Exemples: {ex}"
        )
        # fallback: poids uniforme pour celles non matchées
        w["weight"] = w["weight"].fillna(1.0)

    # agrégation pondérée par date
    # (on normalise les poids par date pour éviter un biais si mismatch)
    def weighted_mean(g: pd.DataFrame) -> pd.Series:
        ww = g["weight"].to_numpy(dtype=float)
        ww = ww / np.sum(ww) if np.sum(ww) > 0 else np.ones_like(ww) / len(ww)
        out = {}
        for v in cfg.weather_vars:
            x = pd.to_numeric(g[v], errors="coerce").to_numpy(dtype=float)
            out[v] = np.nansum(ww * x)
        return pd.Series(out)

    agg = w.groupby("date", sort=True).apply(weighted_mean).reset_index()
    agg = agg.sort_values("date").set_index("date")

    # assurer fréquence horaire (météo est déjà horaire, mais on sécurise)
    agg = agg[~agg.index.duplicated(keep="first")]
    return agg


def _load_consumption(cfg: Config) -> pd.DataFrame:
    """
    Charge conso 2010-2024 depuis data/processed/consommation_YYYY_long.csv
    Puis agrège en horaire pour matcher la météo.
    """
    pattern = os.path.join(cfg.project_root, cfg.consumption_dir, "consommation_*_long.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier conso trouvé avec pattern: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        required = {"datetime", "load_mw"}
        if not required.issubset(set(df.columns)):
            raise ValueError(f"Format conso inattendu: {f} (colonnes: {list(df.columns)})")
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
        df = df.dropna(subset=["datetime", "load_mw"])
        dfs.append(df[["datetime", "load_mw"]])

    c = pd.concat(dfs, ignore_index=True).sort_values("datetime")
    c = c.drop_duplicates(subset=["datetime"], keep="last")
    c = c.set_index("datetime")

    # conso est demi-horaire (00:30, 01:00, ...) -> on agrège par heure
    # choix simple: moyenne des points dans l'heure
    c_hour = c.resample("1H").mean()

    # borne sur la période demandée
    c_hour = c_hour.loc[
        (c_hour.index >= pd.Timestamp(f"{cfg.start_year}-01-01", tz="UTC")) &
        (c_hour.index <= pd.Timestamp(f"{cfg.end_year}-12-31 23:00:00", tz="UTC"))
    ]
    return c_hour


def _make_time_features(idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame(index=idx)
    df["hour"] = idx.hour.astype(int)
    df["dayofweek"] = idx.dayofweek.astype(int)   # 0=Mon
    df["month"] = idx.month.astype(int)
    df["dayofyear"] = idx.dayofyear.astype(int)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    return df


def _make_lag_features(series: pd.Series, lags: Tuple[int, ...]) -> pd.DataFrame:
    out = {}
    for k in lags:
        out[f"load_lag_{k}h"] = series.shift(k)
    return pd.DataFrame(out, index=series.index)


def _make_rolling_features(series: pd.Series, windows: Tuple[int, ...]) -> pd.DataFrame:
    out = {}
    for w in windows:
        out[f"load_rollmean_{w}h"] = series.shift(1).rolling(window=w, min_periods=max(3, w//10)).mean()
        out[f"load_rollstd_{w}h"] = series.shift(1).rolling(window=w, min_periods=max(3, w//10)).std()
    return pd.DataFrame(out, index=series.index)


def _train_val_test_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(cfg.train_end, tz="UTC")
    val_end = pd.Timestamp(cfg.val_end, tz="UTC")
    test_end = pd.Timestamp(cfg.test_end, tz="UTC")

    train = df.loc[:train_end].copy()
    val = df.loc[train_end + pd.Timedelta(hours=1):val_end].copy()
    test = df.loc[val_end + pd.Timedelta(hours=1):test_end].copy()
    return train, val, test

def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)

    # Compat toutes versions sklearn : RMSE = sqrt(MSE)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"MAE": float(mae), "RMSE": rmse, "MAPE_%": mape}



# -----------------------
# MAIN PIPELINE
# -----------------------

def build_dataset(cfg: Config) -> pd.DataFrame:
    xlsx_path = os.path.join(cfg.project_root, cfg.cities_xlsx)
    pop = _safe_read_excel_population(xlsx_path)

    weather = _load_weather(cfg, pop)          # index: date (UTC hourly)
    cons = _load_consumption(cfg)              # index: datetime (UTC hourly)

    # merge sur index horaire
    df = cons.join(weather, how="inner")
    df = df.rename_axis("date").reset_index().set_index("date").sort_index()

    # time features
    tf = _make_time_features(df.index)

    # lags/rolling conso
    lf = _make_lag_features(df["load_mw"], cfg.lags)
    rf = _make_rolling_features(df["load_mw"], cfg.roll_windows)

    # assemble
    full = pd.concat([df, tf, lf, rf], axis=1)

    # drop rows with target missing
    full = full.dropna(subset=["load_mw"])

    # IMPORTANT: on droppe aussi les lignes où les lags sont NA (début de série)
    feature_cols = [c for c in full.columns if c != "load_mw"]
    full = full.dropna(subset=feature_cols)

    return full


def train_model(cfg: Config) -> None:
    os.makedirs(os.path.join(cfg.project_root, cfg.model_dir), exist_ok=True)

    cache_path = os.path.join(cfg.project_root, cfg.dataset_cache_file)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print("✅ Build dataset (hourly, merged conso + météo agrégée + features)...")
    df = build_dataset(cfg)
    print(f"Dataset shape: {df.shape} | from {df.index.min()} to {df.index.max()}")

    # sauvegarde dataset (utile pour notebook)
    try:
        df.reset_index().to_parquet(cache_path, index=False)
        print(f"💾 Dataset sauvegardé: {cache_path}")
    except Exception as e:
        warnings.warn(f"Impossible de sauvegarder en parquet: {e}")

    train, val, test = _train_val_test_split(df, cfg)
    print(f"Split: train={train.shape}, val={val.shape}, test={test.shape}")

    target = "load_mw"
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]
    X_test, y_test = test.drop(columns=[target]), test[target]

    # colonnes catégorielles (calendaires) vs numériques
    cat_cols = ["hour", "dayofweek", "month", "is_weekend"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), num_cols),
        ],
        remainder="drop"
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=None,
        max_iter=600,
        max_leaf_nodes=64,
        min_samples_leaf=30,
        l2_regularization=0.0,
        random_state=cfg.random_state,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preproc),
        ("model", model),
    ])

    print("🚀 Training model...")
    pipe.fit(X_train, y_train)

    # évaluation
    pred_val = pipe.predict(X_val)
    pred_test = pipe.predict(X_test)

    print("\n📊 Metrics")
    print("Validation:", _metrics(y_val.values, pred_val))
    print("Test      :", _metrics(y_test.values, pred_test))

    # baseline naive: y(t) = y(t-24h)
    if "load_lag_24h" in X_test.columns:
        baseline = X_test["load_lag_24h"].values
        print("Baseline (lag 24h) Test:", _metrics(y_test.values, baseline))

    # save
    model_path = os.path.join(cfg.project_root, cfg.model_dir, cfg.model_file)
    joblib.dump(pipe, model_path)
    print(f"\n✅ Model saved: {model_path}")


if __name__ == "__main__":
    warnings.filterwarnings("default")
    cfg = Config()
    train_model(cfg)
