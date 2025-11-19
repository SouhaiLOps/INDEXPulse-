# src/preprocessing.py

from __future__ import annotations

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Iterable, Dict

# ====== CONFIG DE BASE ======

TICKER_CAC40 = "^FCHI"

# fenêtres & lags
FE_WINDOWS = [5, 10, 20, 50, 100, 200]
FE_LAGS = [1, 2, 5, 10]
HORIZON_D = 1  # prédire le retour à J+1

DEFAULT_YEARS = 10
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"


# ====== 1. DOWNLOAD & MISE EN FORME OHLCV ======

def download_ohlcv(
    tickers: Iterable[str],
    years: int = DEFAULT_YEARS,
    raw_dir: str = RAW_DIR,
) -> pd.DataFrame:
    """
    Télécharge les données OHLCV quotidiennes sur 'years' années
    pour une liste de tickers Yahoo, retourne un DataFrame 'tidy' :
    colonnes = [date, ticker, open, high, low, close, adj_close, volume]
    """
    tickers = list(tickers)
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)

    raw = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )

    # raw: colonnes multi-index (niveau 0 = ticker, niveau 1 = champs)
    tidy = (
        raw.stack(level=0)
           .rename_axis(index=["date", "ticker"])
           .reset_index()
           .rename(columns={
               "Open": "open",
               "High": "high",
               "Low": "low",
               "Close": "close",
               "Adj Close": "adj_close",
               "Volume": "volume",
           })
           .sort_values(["ticker", "date"])
           .reset_index(drop=True)
    )

    # sauvegarde brute (optionnel mais utile pour debug)
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(Path(raw_dir) / "ohlcv_full.parquet", index=False)

    return tidy


# ====== 2. FEATURE ENGINEERING ======

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


def build_features_one_ticker(tidy_one: pd.DataFrame, horizon: int = HORIZON_D) -> pd.DataFrame:
    """
    tidy_one : données OHLCV pour un seul ticker
               colonnes = [date,ticker,open,high,low,close,adj_close,volume]
    Retour : DataFrame features + cibles, lignes datées, sans NaN critiques.
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


# ====== 3. PIPELINE CAC40 ======

def preprocess_cac40(
    years: int = DEFAULT_YEARS,
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROC_DIR,
) -> pd.DataFrame:
    """
    Pipeline complet pour le CAC40 :
    - téléchargement OHLCV (10 ans par défaut)
    - feature engineering
    - sauvegarde dans data/processed
    - retourne le DataFrame de features
    """
    tidy = download_ohlcv([TICKER_CAC40], years=years, raw_dir=raw_dir)
    one = tidy[tidy["ticker"] == TICKER_CAC40].copy()
    feat = build_features_one_ticker(one, horizon=HORIZON_D)

    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(processed_dir) / f"FCHI_features_h{HORIZON_D}d.parquet"
    feat.to_parquet(out_path, index=False)

    return feat
