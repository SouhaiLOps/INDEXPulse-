# src/preprocessing.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# -----------------------------------------------------
# Chemins basés sur l'emplacement du fichier
# -----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Fenêtres & lags
FE_WINDOWS = [5, 10, 20, 50, 100, 200]
FE_LAGS = [1, 2, 5, 10]
HORIZON_D = 1  # prédire le retour à J+1

TICKER_CAC40 = "^FCHI"



# ========= FONCTIONS DE FEATURES =========

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    di = df["date"]
    df["dow"] = di.dt.dayofweek
    df["month"] = di.dt.month
    df["is_month_end"] = di.dt.is_month_end.astype(int)
    df["is_quarter_end"] = di.dt.is_quarter_end.astype(int)
    iso = di.dt.isocalendar()
    df["weekofyear"] = iso.week.astype(int)
    df["dayofyear"] = di.dt.dayofyear
    return df


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df["ret1"] = df["close"].pct_change()
    df["logret1"] = np.log(df["close"] / df["close"].shift(1))
    for L in FE_LAGS:
        df[f"ret_lag{L}"] = df["ret1"].shift(L)
        df[f"logret_lag{L}"] = df["logret1"].shift(L)
    return df


def add_ma_vol_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in FE_WINDOWS:
        df[f"sma{w}"] = df["close"].rolling(w, min_periods=w).mean()
        df[f"ema{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
        df[f"volstd{w}"] = df["ret1"].rolling(w, min_periods=w).std()
        df[f"vol_sma{w}"] = df["volume"].rolling(w, min_periods=w).mean()
    for L in FE_LAGS:
        df[f"vol_lag{L}"] = df["volume"].shift(L)
    return df


def add_atr(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(n, min_periods=n).mean()
    return df


def make_target(df: pd.DataFrame, horizon: int = HORIZON_D) -> pd.DataFrame:
    df["target_ret"] = df["close"].pct_change(horizon).shift(-horizon)
    df["target_logret"] = (
        np.log(df["close"] / df["close"].shift(horizon))
    ).shift(-horizon)
    return df


def build_features_one_ticker(tidy_one: pd.DataFrame,
                              horizon: int = HORIZON_D) -> pd.DataFrame:
    """
    tidy_one : DataFrame OHLCV pour un seul ticker :
      [date, ticker, open, high, low, close, adj_close, volume]
    Retour : features + target pour ce ticker.
    """
    df = tidy_one.sort_values("date").copy()

    df = add_calendar_features(df)
    df = add_return_features(df)
    df = add_ma_vol_features(df)
    df = add_atr(df, n=14)
    df = make_target(df, horizon=horizon)

    need_cols = ["ret1", "logret1", "atr", "target_ret"]
    need_cols += [f"sma{w}" for w in FE_WINDOWS] + [f"volstd{w}" for w in FE_WINDOWS]
    df = df.dropna(subset=need_cols).reset_index(drop=True)

    return df


# ========= PIPELINE CAC40 SANS DOWNLOAD =========

def preprocess_cac40_from_tidy(
    tidy: pd.DataFrame,
    processed_dir: Path | str = PROC_DIR,
) -> pd.DataFrame:
    """
    Applique le feature engineering au tidy du CAC40 et sauvegarde
    le dataset dans data/processed.
    """
    one = tidy[tidy["ticker"] == TICKER_CAC40].copy()
    feat = build_features_one_ticker(one, horizon=HORIZON_D)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / f"FCHI_features_h{HORIZON_D}d.parquet"
    feat.to_parquet(out_path, index=False)
    print("[OK] Features FCHI écrites dans :", out_path)

    return feat

