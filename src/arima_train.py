# src/arima_train.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# -----------------------------------------------------
# Chemins basés sur l'arborescence du projet
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "FCHI_features_h1d.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "arima_logret1_FCHI.pkl"


# -----------------------------------------------------
# Utilitaires
# -----------------------------------------------------

def eval_metrics(y_true: pd.Series, y_pred: pd.Series, name: str = ""):
    """Calcule RMSE et MAE et les affiche."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    print(f"{name} RMSE={rmse:.6f}   MAE={mae:.6f}")
    return rmse, mae


def load_features() -> pd.DataFrame:
    """Charge le parquet FCHI_features_h1d.parquet."""
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Fichier de features introuvable : {PROCESSED_PATH}\n"
            "Lance d'abord le preprocessing pour le générer."
        )

    df = pd.read_parquet(PROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame):
    """
    Découpe temporel : 70% train, 15% val, 15% test
    en respectant l’ordre des dates.
    """
    df = df.sort_values("date").copy()

    unique_dates = df["date"].sort_values().unique()
    n_dates = len(unique_dates)

    train_end = unique_dates[int(0.70 * n_dates)]
    val_end = unique_dates[int(0.85 * n_dates)]

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[df["date"] > val_end]

    print(
        f"Split temporel : train={len(train)}, val={len(val)}, "
        f"test={len(test)}, n_dates={n_dates}"
    )
    return train, val, test, train_end, val_end


# -----------------------------------------------------
# Entraînement ARIMA sur logret1
# -----------------------------------------------------

def train_arima_logret1(order=(1, 0, 1)):
    """
    - charge FCHI_features_h1d.parquet
    - split train/val/test par date
    - ARIMA sur logret1 (univarié)
    - affiche les métriques sur val & test
    - sauvegarde le modèle refitté train+val dans models/
    """
    df = load_features()

    train_df, val_df, test_df, train_end, val_end = time_split(df)

    # Série logret1 indexée par date
    series = df.set_index("date")["logret1"].sort_index()

    train_s = series[series.index <= train_end].dropna()
    val_s = series[(series.index > train_end) & (series.index <= val_end)].dropna()
    test_s = series[series.index > val_end].dropna()

    print(
        f"Séries : train={len(train_s)}, val={len(val_s)}, "
        f"test={len(test_s)}"
    )

    # ---------- 1) Fit sur train, prévision sur val ----------
    arima_model = ARIMA(train_s, order=order)
    arima_res = arima_model.fit()

    fc_val = arima_res.forecast(steps=len(val_s))
    fc_val.index = val_s.index  # aligner les index

    rmse_val, mae_val = eval_metrics(val_s, fc_val, "ARIMA VAL")

    # ---------- 2) Refit sur train+val, prévision sur test ----------
    trainval_s = series[series.index <= val_end].dropna()

    arima_model_tv = ARIMA(trainval_s, order=order)
    arima_res_tv = arima_model_tv.fit()

    fc_test = arima_res_tv.forecast(steps=len(test_s))
    fc_test.index = test_s.index

    rmse_test, mae_test = eval_metrics(test_s, fc_test, "ARIMA TEST")

    # ---------- 3) Sauvegarde du modèle ----------
    arima_res_tv.save(MODEL_PATH)
    print(f"Modèle ARIMA sauvegardé sous : {MODEL_PATH}")

    return {
        "rmse_val": rmse_val,
        "mae_val": mae_val,
        "rmse_test": rmse_test,
        "mae_test": mae_test,
        "model_path": MODEL_PATH,
    }


# -----------------------------------------------------
# Point d’entrée script
# -----------------------------------------------------

def main():
    metrics = train_arima_logret1(order=(1, 0, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
