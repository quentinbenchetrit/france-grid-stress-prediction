#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = "/home/onyxia/work/france-grid-stress-prediction"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hgb_load_forecast.joblib")
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dataset_model_hourly.parquet")

OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "backtest_jplus1_2023_predictions.csv")


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def main():
    # --- checks
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}\n"
            "Tu dois d'abord exécuter train_xgboost.py (il génère dataset_model_hourly.parquet)."
        )

    print("✅ Loading model:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    print("✅ Loading dataset:", DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)

    # Le parquet a été sauvegardé via df.reset_index() -> on s'attend à une colonne 'date'
    if "date" not in df.columns:
        raise ValueError(f"Colonne 'date' introuvable dans {DATASET_PATH}. Colonnes: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").set_index("date")

    if "load_mw" not in df.columns:
        raise ValueError("Colonne 'load_mw' introuvable dans le dataset.")

    # --- Build J+1 target (t+24h)
    H = 24
    df["y_true_tplus24"] = df["load_mw"].shift(-H)

    # On prédit y(t+24) en utilisant les features au temps t
    # Donc on prend X à t, y_true à t+24
    feature_cols = [c for c in df.columns if c not in ["load_mw", "y_true_tplus24"]]

    # --- Filtrage période 2023 (features prises en 2023, cible en 2023 aussi)
    # Pour avoir y_true_tplus24 dans 2023, il faut que date <= 2023-12-31 23:00 - 24h
    start = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2023-12-31 23:00:00", tz="UTC") - pd.Timedelta(hours=H)

    bt = df.loc[start:end].copy()
    bt = bt.dropna(subset=feature_cols + ["y_true_tplus24"])

    if bt.empty:
        raise ValueError("Backtest 2023 vide après filtrage (vérifie que le dataset contient bien 2023).")

    X = bt[feature_cols]
    y_true = bt["y_true_tplus24"].values

    print(f"✅ Backtest window: {bt.index.min()} -> {bt.index.max()} ({len(bt):,} heures)")

    # --- Predict
    y_pred = model.predict(X)

    # --- Report metrics global
    global_metrics = metrics(y_true, y_pred)
    print("\n📊 Global J+1 metrics on 2023 (hourly):")
    print(global_metrics)

    # --- Assemble output with the target timestamp (t+24)
    out = pd.DataFrame({
        "t_features": bt.index,
        "t_target": bt.index + pd.Timedelta(hours=H),
        "y_true_load_mw": y_true,
        "y_pred_load_mw": y_pred,
    })
    out["abs_error_mw"] = np.abs(out["y_true_load_mw"] - out["y_pred_load_mw"])
    out["pct_error_%"] = out["abs_error_mw"] / np.clip(np.abs(out["y_true_load_mw"]), 1e-6, None) * 100

    # --- Daily % error (moyenne par jour sur les 24 prédictions)
    out["day"] = out["t_target"].dt.floor("D")
    daily = out.groupby("day").agg(
        daily_mape_pct=("pct_error_%", "mean"),
        daily_mae_mw=("abs_error_mw", "mean"),
        daily_rmse_mw=("abs_error_mw", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        n=("pct_error_%", "size"),
    ).reset_index()

    print("\n📆 Daily error summary (2023):")
    print("Mean daily MAPE %:", float(daily["daily_mape_pct"].mean()))
    print("Median daily MAPE %:", float(daily["daily_mape_pct"].median()))
    print("Worst day MAPE %:", float(daily["daily_mape_pct"].max()), "on", daily.loc[daily["daily_mape_pct"].idxmax(), "day"])

    # --- Save
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    daily.to_csv(OUT_CSV.replace(".csv", "_daily.csv"), index=False)

    print(f"\n💾 Saved hourly backtest to: {OUT_CSV}")
    print(f"💾 Saved daily summary to: {OUT_CSV.replace('.csv', '_daily.csv')}")


if __name__ == "__main__":
    main()
