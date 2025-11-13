from __future__ import annotations
import pandas as pd, numpy as np
from typing import Iterable

def _feat_one_series(s: pd.Series,
                     windows: Iterable[int],
                     lags: Iterable[int]) -> pd.DataFrame:
    s = s.dropna().rename("close")
    df = pd.DataFrame({"close": s})
    # rendements
    df["ret1"] = df["close"].pct_change()
    df["logret1"] = np.log(df["close"] / df["close"].shift(1))
    # moyennes mobiles / volatilité
    for w in windows:
        df[f"sma{w}"] = df["close"].rolling(w).mean()
        df[f"ema{w}"] = df["close"].ewm(span=w, adjust=False).mean()
        df[f"vol{w}"] = df["ret1"].rolling(w).std()
    # décalages de rendements (lags)
    for L in lags:
        df[f"ret_lag{L}"] = df["ret1"].shift(L)
    return df

def make_dataset(close_wide: pd.DataFrame,
                 windows: Iterable[int],
                 lags: Iterable[int],
                 horizon_days: int) -> pd.DataFrame:
    """
    close_wide: colonnes = tickers, index = dates
    Retourne un dataset long avec features + target (retour futur).
    """
    frames = []
    for col in close_wide.columns:
        f = _feat_one_series(close_wide[col], windows, lags)
        # cible = rendement futur à horizon h
        f["target"] = f["close"].pct_change(horizon_days).shift(-horizon_days)
        f["ticker"] = col
        frames.append(f)
    data = pd.concat(frames).reset_index().rename(columns={"index":"date"})
    # nettoyage
    data = data.dropna().sort_values(["ticker","date"])
    # normalisation simple par ticker (optionnel)
    for c in [c for c in data.columns if c not in ("date","ticker","target","close")]:
        data[c] = data.groupby("ticker")[c].transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return data
