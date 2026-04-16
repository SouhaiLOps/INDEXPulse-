# src/mmm_features.py
# PortfolioMMM — Feature engineering inspiré du Marketing Mix Modelling
# =====================================================================
#
# Analogie MMM → Finance :
#
#   ADSTOCK     EMA des rendements passés avec demi-vie par actif
#               → "Combien de jours un choc de rendement persiste ?"
#               → Or : mémoire longue (10j) / BTC : mémoire courte (3j)
#
#   SATURATION  Sigmoïde des rendements cumulés sur 20 jours
#               → "L'effet marginal diminue quand l'actif a déjà
#                  beaucoup bougé" (marché suracheté/survendu)
#               → Analogue : rendement décroissant d'un canal média
#
#   FEATURES    Volatilité, momentum, lags, SMA/EMA — features
#   TEMPORELLES classiques qui capturent la dynamique de prix
#
#   TARGET      Rendement futur à J+HORIZON_D pour chaque actif
#               → Ce que le modèle bayésien cherchera à expliquer
#
# Input  : close_wide  DataFrame (date × ticker) — prix de clôture
# Output : dataset     DataFrame (date × features+targets) — zéro NaN
# =====================================================================

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Import config — fonctionne aussi si appelé en standalone
try:
    from src.config import (
        TICKERS, ADSTOCK_HALFLIFE, SATURATION_SCALE,
        FEATURE_WINDOWS, FEATURE_LAGS, HORIZON_D, PROC_DIR,
    )
except ImportError:
    # Fallback pour tests directs
    TICKERS          = ["GC=F", "CL=F", "EURUSD=X", "BTC-USD"]
    ADSTOCK_HALFLIFE = {"GC=F": 10, "CL=F": 5, "EURUSD=X": 7, "BTC-USD": 3}
    SATURATION_SCALE = {"GC=F": 0.05, "CL=F": 0.10, "EURUSD=X": 0.03, "BTC-USD": 0.20}
    FEATURE_WINDOWS  = [5, 20, 60]
    FEATURE_LAGS     = [1, 2, 5]
    HORIZON_D        = 1
    PROC_DIR         = "data/processed"


# ── 1. Adstock ────────────────────────────────────────────────────────────────

def compute_adstock(series: pd.Series, halflife: float) -> pd.Series:
    """
    EMA avec demi-vie explicite en jours.

    alpha = 1 - exp(-ln(2) / halflife)

    Interprétation : après `halflife` jours, un choc de rendement
    a perdu exactement 50% de son influence sur l'adstock.
    Plus halflife est grand, plus la mémoire est longue.
    """
    if halflife <= 0:
        raise ValueError(f"halflife doit être > 0, reçu {halflife}")
    alpha = 1 - np.exp(-np.log(2) / halflife)
    return series.ewm(alpha=alpha, adjust=False).mean()


def add_adstock_features(rets: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule l'adstock pour chaque actif avec sa demi-vie spécifique.

    Retourne un DataFrame avec colonnes : adstock_{ticker}
    """
    result = pd.DataFrame(index=rets.index)
    for t in TICKERS:
        if t not in rets.columns:
            continue
        h = ADSTOCK_HALFLIFE[t]
        result[f"adstock_{t}"] = compute_adstock(rets[t].fillna(0), h)
    return result


# ── 2. Saturation ─────────────────────────────────────────────────────────────

def saturation_transform(series: pd.Series, scale: float) -> pd.Series:
    """
    Sigmoïde symétrique centrée sur 0 :
        sat(x) = 2 / (1 + exp(-x / scale)) - 1   ∈ (-1, +1)

    `scale` = point d'inflexion = rendement cumulé à partir duquel
    l'effet marginal commence à décroître significativement.

    Exemples :
        sat(0)      = 0.0   (neutre)
        sat(scale)  ≈ 0.46  (effet notable mais pas saturé)
        sat(3*scale)≈ 0.90  (quasi-saturé)
    """
    return 2 / (1 + np.exp(-series / scale)) - 1


def add_saturation_features(
    rets: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """
    Calcule la saturation sur le rendement cumulé glissant de `window` jours.

    Retourne un DataFrame avec colonnes : sat_{ticker}
    """
    result = pd.DataFrame(index=rets.index)
    for t in TICKERS:
        if t not in rets.columns:
            continue
        scale      = SATURATION_SCALE[t]
        cum_ret    = rets[t].fillna(0).rolling(window).sum()
        result[f"sat_{t}"] = saturation_transform(cum_ret, scale)
    return result


# ── 3. Features temporelles ───────────────────────────────────────────────────

def add_temporal_features(
    close_wide: pd.DataFrame,
    rets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Features classiques de séries temporelles financières :
      - vol{w}_{t}     : volatilité réalisée sur w jours (std des rendements)
      - mom{w}_{t}     : momentum sur w jours (moyenne des rendements)
      - sma{w}_{t}     : SMA normalisée (close / SMA - 1)
      - ema{w}_{t}     : EMA normalisée (close / EMA - 1)
      - ret_lag{L}_{t} : rendement décalé de L jours (autocorrélation)
      - logret_{t}     : log-rendement J-1

    La normalisation SMA/EMA (close/MA - 1) rend la feature
    stationnaire et comparable entre actifs de niveaux différents.
    """
    result = pd.DataFrame(index=close_wide.index)

    for t in TICKERS:
        if t not in close_wide.columns:
            continue

        price = close_wide[t]
        ret   = rets[t].fillna(0)

        # Log-rendement
        result[f"logret_{t}"] = np.log(price / price.shift(1))

        # Volatilité, momentum, SMA/EMA normalisées
        for w in FEATURE_WINDOWS:
            result[f"vol{w}_{t}"]  = ret.rolling(w).std()
            result[f"mom{w}_{t}"]  = ret.rolling(w).mean()
            sma = price.rolling(w).mean()
            ema = price.ewm(span=w, adjust=False).mean()
            result[f"sma{w}_{t}"]  = (price / sma) - 1
            result[f"ema{w}_{t}"]  = (price / ema) - 1

        # Lags de rendements
        for L in FEATURE_LAGS:
            result[f"ret_lag{L}_{t}"] = ret.shift(L)

    return result


# ── 4. Features calendaires ───────────────────────────────────────────────────

def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Features temporelles cycliques encodées en sin/cos pour que
    lundi et vendredi soient "proches" dans l'espace des features.

    Colonnes générées :
      - dow_sin / dow_cos        : jour de la semaine (0=lundi … 4=vendredi)
      - month_sin / month_cos    : mois de l'année
      - is_monday / is_friday    : effets de début/fin de semaine
      - is_month_end             : effet fin de mois (rebalancement)
    """
    result = pd.DataFrame(index=index)
    dow   = index.dayofweek
    month = index.month

    result["dow_sin"]      = np.sin(2 * np.pi * dow / 5)
    result["dow_cos"]      = np.cos(2 * np.pi * dow / 5)
    result["month_sin"]    = np.sin(2 * np.pi * month / 12)
    result["month_cos"]    = np.cos(2 * np.pi * month / 12)
    result["is_monday"]    = (dow == 0).astype(int)
    result["is_friday"]    = (dow == 4).astype(int)
    result["is_month_end"] = index.is_month_end.astype(int)
    return result


# ── 5. Target ─────────────────────────────────────────────────────────────────

def add_targets(
    rets: pd.DataFrame,
    horizon: int = HORIZON_D,
) -> pd.DataFrame:
    """
    Rendement futur à +horizon jours pour chaque actif.

    C'est la variable que le modèle bayésien cherche à expliquer.
    Le décalage négatif (shift(-horizon)) est la cible "future".
    """
    result = pd.DataFrame(index=rets.index)
    for t in TICKERS:
        if t not in rets.columns:
            continue
        result[f"target_{t}"] = rets[t].shift(-horizon)
    return result


# ── 6. Pipeline principal ─────────────────────────────────────────────────────

def build_mmm_features(
    close_wide: pd.DataFrame,
    horizon: int = HORIZON_D,
    saturation_window: int = 20,
    save: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet : close_wide → dataset features + targets.

    Paramètres
    ----------
    close_wide        : DataFrame wide (date × ticker), prix de clôture
    horizon           : nombre de jours pour la target future
    saturation_window : fenêtre du rendement cumulé pour la saturation
    save              : si True, sauvegarde en parquet dans PROC_DIR

    Retourne
    --------
    dataset : DataFrame (date × features + targets)
              index = date, zéro NaN, prêt pour bayes_allocation.py
    """
    print(f"[mmm_features] Tickers : {list(close_wide.columns)}")
    print(f"[mmm_features] Période : {close_wide.index[0].date()} → {close_wide.index[-1].date()}")
    print(f"[mmm_features] Lignes brutes : {len(close_wide)}")

    # Rendements (base de tout)
    rets = close_wide.pct_change()

    # Construction des blocs de features
    adstock_feats  = add_adstock_features(rets)
    sat_feats      = add_saturation_features(rets, window=saturation_window)
    temporal_feats = add_temporal_features(close_wide, rets)
    calendar_feats = add_calendar_features(close_wide.index)
    targets        = add_targets(rets, horizon=horizon)

    # Assemblage
    dataset = pd.concat(
        [adstock_feats, sat_feats, temporal_feats, calendar_feats, targets],
        axis=1,
    )
    dataset.index.name = "date"

    # Suppression des NaN (fenêtres roulantes + shift target)
    n_before = len(dataset)
    dataset  = dataset.dropna()
    n_after  = len(dataset)
    n_lost   = n_before - n_after

    # Colonnes features / targets
    feat_cols   = [c for c in dataset.columns if not c.startswith("target_")]
    target_cols = [c for c in dataset.columns if c.startswith("target_")]

    print(f"[mmm_features] Lignes finales : {n_after} (perdues : {n_lost})")
    print(f"[mmm_features] Features       : {len(feat_cols)}")
    print(f"[mmm_features]   → adstock    : {sum(1 for c in feat_cols if 'adstock' in c)}")
    print(f"[mmm_features]   → saturation : {sum(1 for c in feat_cols if 'sat_' in c)}")
    print(f"[mmm_features]   → temporelles: {sum(1 for c in feat_cols if 'adstock' not in c and 'sat_' not in c)}")
    print(f"[mmm_features] Targets        : {len(target_cols)} ({target_cols})")
    print(f"[mmm_features] NaN restants   : {dataset.isna().sum().sum()}")

    if save:
        out = Path(PROC_DIR) / f"mmm_features_h{horizon}d.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(out, index=True)
        print(f"[mmm_features] Sauvegardé → {out}")

    return dataset


# ── 7. Helpers d'inspection ───────────────────────────────────────────────────

def feature_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Résumé statistique des features groupées par type et par actif.
    Utile pour vérifier que les features sont bien distribuées.
    """
    feat_cols = [c for c in dataset.columns if not c.startswith("target_")]
    summary   = dataset[feat_cols].describe().T
    summary["feature_type"] = summary.index.map(lambda c:
        "adstock"    if c.startswith("adstock") else
        "saturation" if c.startswith("sat_")    else
        "calendar"   if any(k in c for k in ["dow", "month", "is_"]) else
        "temporal"
    )
    return summary[["feature_type", "mean", "std", "min", "max"]]


def get_feature_groups(dataset: pd.DataFrame) -> dict:
    """
    Retourne un dict {groupe: [colonnes]} pour usage dans bayes_allocation.py.

    Structure :
        {
            "adstock":    ["adstock_GC=F", ...],
            "saturation": ["sat_GC=F", ...],
            "temporal":   ["vol5_GC=F", ...],
            "calendar":   ["dow_sin", ...],
            "targets":    ["target_GC=F", ...],
        }
    """
    cols = dataset.columns.tolist()
    return {
        "adstock":    [c for c in cols if c.startswith("adstock_")],
        "saturation": [c for c in cols if c.startswith("sat_")],
        "temporal":   [c for c in cols if not any(
                           c.startswith(p) for p in
                           ["adstock_", "sat_", "dow", "month", "is_", "target_"])],
        "calendar":   [c for c in cols if any(
                           k in c for k in ["dow", "month", "is_"])],
        "targets":    [c for c in cols if c.startswith("target_")],
    }


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.download_yf import fetch_close

    print("=== PortfolioMMM — Construction des features ===\n")
    close_wide = fetch_close()
    dataset    = build_mmm_features(close_wide, horizon=HORIZON_D, save=True)

    print("\n=== Résumé par groupe ===")
    groups = get_feature_groups(dataset)
    for g, cols in groups.items():
        print(f"  {g:<12} : {len(cols):>3} colonnes")

    print("\n=== Aperçu des 3 premières features de chaque groupe ===")
    for g, cols in groups.items():
        if g == "targets":
            continue
        sample = dataset[cols[:3]].tail(3).round(4)
        print(f"\n-- {g} --")
        print(sample.to_string())