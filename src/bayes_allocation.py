# src/bayes_allocation.py
# PortfolioMMM — Régression bayésienne & allocation de portefeuille
# =================================================================
#
# Logique MMM appliquée à la finance :
#
#   rendement_portfolio(t) = α
#                          + Σ β_adstock[i]  × adstock[i](t)
#                          + Σ β_sat[i]      × saturation[i](t)
#                          + Σ β_temp[i]     × features_temporelles[i](t)
#                          + ε(t)
#
#   Les bêtas estimés → contribution de chaque actif
#   Softmax(contributions) → poids d'allocation
#   Distribution postérieure → intervalles de confiance sur les poids
#
# =================================================================

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from src.config import TICKERS, TICKER_LABELS, PROC_DIR
except ImportError:
    TICKERS      = ["GC=F", "CL=F", "EURUSD=X", "BTC-USD"]
    TICKER_LABELS = {
        "GC=F": "Or", "CL=F": "Pétrole WTI",
        "EURUSD=X": "EUR/USD", "BTC-USD": "Bitcoin",
    }
    PROC_DIR = "data/processed"


# ── 1. Préparation des données ────────────────────────────────────────────────

def prepare_Xy(
    dataset: pd.DataFrame,
    target_mode: str = "portfolio",
    portfolio_weights: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extrait X (features) et y (target) depuis le dataset mmm_features.

    Paramètres
    ----------
    dataset          : output de build_mmm_features()
    target_mode      : "portfolio"  → y = rendement du portefeuille équipondéré
                       "per_asset"  → y = rendement moyen de tous les actifs
                       "custom"     → y = somme pondérée selon portfolio_weights
    portfolio_weights: dict {ticker: poids} si target_mode="custom"

    Retourne
    --------
    X          : array (N, P) — features normalisées
    y          : array (N,)   — target
    feat_cols  : liste des noms de colonnes de X
    """
    target_cols = [c for c in dataset.columns if c.startswith("target_")]
    feat_cols   = [c for c in dataset.columns if not c.startswith("target_")]

    # Garder uniquement les features MMM clés pour le modèle bayésien
    # (adstock + saturation + logret + quelques features temporelles clés)
    # Les features complètes vont dans bayes_allocation mais on sélectionne
    # les plus informatives pour garder le modèle interprétable
    key_prefixes = ("adstock_", "sat_", "logret_", "vol5_", "vol20_",
                    "mom5_", "mom20_", "ret_lag1_", "ret_lag2_",
                    "dow_sin", "dow_cos", "month_sin", "month_cos",
                    "is_monday", "is_friday", "is_month_end")
    feat_cols = [c for c in feat_cols
                 if any(c.startswith(p) or c == p for p in key_prefixes)]

    X = dataset[feat_cols].values.astype(float)

    # Normalisation Z-score par colonne (important pour les priors bayésiens)
    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-9
    X_norm = (X - X_mean) / X_std

    # Construction de la target
    targets = dataset[target_cols]
    if target_mode == "portfolio":
        y = targets.mean(axis=1).values   # équipondéré
    elif target_mode == "custom" and portfolio_weights:
        y = sum(
            portfolio_weights.get(t, 0.25) * targets[f"target_{t}"]
            for t in TICKERS if f"target_{t}" in targets.columns
        ).values
    else:
        y = targets.mean(axis=1).values

    return X_norm, y, feat_cols, X_mean, X_std


# ── 2. Modèle bayésien ────────────────────────────────────────────────────────

def build_bayesian_model(
    X: np.ndarray,
    y: np.ndarray,
    feat_cols: list[str],
    prior_sigma: float = 0.1,
) -> pm.Model:
    """
    Régression bayésienne linéaire avec priors régularisants.

    Prior sur β ~ Normal(0, prior_sigma)
        → Analogue Ridge : on croit que les effets sont petits
          sauf preuve contraire dans les données
        → prior_sigma = 0.1 est conservateur (features normalisées)

    Prior sur σ ~ HalfNormal(0.02)
        → Bruit attendu faible pour des rendements quotidiens
    """
    n_features = X.shape[1]

    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=0.01)
        beta  = pm.Normal("beta",  mu=0, sigma=prior_sigma,
                          shape=n_features)
        sigma = pm.HalfNormal("sigma", sigma=0.02)

        # Vraisemblance
        mu = alpha + pm.math.dot(X, beta)
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    return model


def run_mcmc(
    model: pm.Model,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """Lance le sampling NUTS et retourne la trace postérieure."""
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )
    return trace


# ── 3. Extraction des contributions par actif ─────────────────────────────────

def extract_asset_contributions(
    trace: az.InferenceData,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Calcule la contribution nette de chaque actif à partir des bêtas.

    Pour chaque actif, on somme les bêtas de toutes ses features
    (adstock, saturation, logret, vol, mom, lags).
    Cela donne une "valeur canal" analogue au MMM.

    Retourne un DataFrame :
        ticker | contrib_mean | contrib_std | contrib_lo | contrib_hi
    """
    beta_samples = trace.posterior["beta"].values
    # shape : (chains, draws, n_features) → (samples, n_features)
    beta_flat = beta_samples.reshape(-1, len(feat_cols))

    rows = []
    for t in TICKERS:
        # Indices des features appartenant à cet actif
        idx = [i for i, c in enumerate(feat_cols)
               if t in c or c.replace("=", "_").split("_")[-1] in t]
        if not idx:
            continue

        # Contribution = somme des bêtas de cet actif sur tous les samples
        contrib_samples = beta_flat[:, idx].sum(axis=1)

        rows.append({
            "ticker":       t,
            "label":        TICKER_LABELS.get(t, t),
            "contrib_mean": contrib_samples.mean(),
            "contrib_std":  contrib_samples.std(),
            "contrib_lo":   np.percentile(contrib_samples, 5),
            "contrib_hi":   np.percentile(contrib_samples, 95),
        })

    return pd.DataFrame(rows).set_index("ticker")


# ── 4. Allocation — contributions → poids ─────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def compute_allocation(
    contributions: pd.DataFrame,
    trace: az.InferenceData,
    feat_cols: list[str],
) -> pd.DataFrame:
    """
    Convertit les contributions bayésiennes en poids d'allocation.

    Softmax(contributions) → poids positifs qui somment à 1.

    Retourne un DataFrame avec les poids et leurs intervalles de confiance :
        ticker | label | weight | weight_lo | weight_hi | contrib_mean
    """
    # Poids ponctuel (moyenne postérieure)
    contrib_means = contributions["contrib_mean"].values
    weights_mean  = softmax(contrib_means)

    # Distribution des poids (incertitude bayésienne)
    beta_samples = trace.posterior["beta"].values.reshape(-1, len(feat_cols))
    all_weights  = []

    for sample in beta_samples:
        contribs = []
        for t in TICKERS:
            idx = [i for i, c in enumerate(feat_cols)
                   if t in c or c.replace("=", "_").split("_")[-1] in t]
            contribs.append(sample[idx].sum() if idx else 0.0)
        all_weights.append(softmax(np.array(contribs)))

    all_weights = np.array(all_weights)   # (n_samples, n_tickers)

    allocation = pd.DataFrame({
        "ticker":    TICKERS,
        "label":     [TICKER_LABELS.get(t, t) for t in TICKERS],
        "weight":    weights_mean,
        "weight_lo": np.percentile(all_weights, 5, axis=0),
        "weight_hi": np.percentile(all_weights, 95, axis=0),
        "contrib_mean": contributions["contrib_mean"].values,
    }).set_index("ticker")

    return allocation


# ── 5. Diagnostic du modèle ───────────────────────────────────────────────────

def model_diagnostics(trace: az.InferenceData, feat_cols: list[str]) -> dict:
    """
    Retourne les indicateurs de qualité du modèle MCMC :
      - R-hat  : convergence des chaînes (< 1.01 = bon)
      - ESS    : taille effective de l'échantillon (> 400 = bon)
      - LOO    : Leave-One-Out cross-validation score (prédictif)
    """
    summary = az.summary(trace, var_names=["beta", "alpha", "sigma"],
                         round_to=4)
    rhat_max = summary["r_hat"].max()
    ess_min  = summary["ess_bulk"].min()

    diag = {
        "r_hat_max":  rhat_max,
        "ess_min":    ess_min,
        "converged":  rhat_max < 1.05 and ess_min > 200,
        "n_features": len(feat_cols),
        "n_samples":  int(trace.posterior["beta"].values.reshape(-1, len(feat_cols)).shape[0]),
    }

    # LOO (si possible — peut être lent sur grands datasets)
    try:
        loo = az.loo(trace)
        diag["loo_score"] = float(loo.elpd_loo)
    except Exception:
        diag["loo_score"] = None

    return diag


# ── 6. Pipeline principal ─────────────────────────────────────────────────────

def run_portfolio_mmm(
    dataset: pd.DataFrame,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 2,
    save: bool = True,
    output_dir: str = PROC_DIR,
) -> dict:
    """
    Pipeline complet : dataset mmm_features → allocation bayésienne.

    Retourne un dict avec :
        allocation   : DataFrame poids par actif avec IC
        contributions: DataFrame contributions brutes
        diagnostics  : dict qualité MCMC
        trace        : az.InferenceData (posterior complet)
    """
    print("=" * 60)
    print("  PortfolioMMM — Régression bayésienne")
    print("=" * 60)

    # 1. Préparer X et y
    print("\n[1/4] Préparation des données…")
    X, y, feat_cols, X_mean, X_std = prepare_Xy(dataset)
    print(f"      X shape    : {X.shape}")
    print(f"      y shape    : {y.shape}")
    print(f"      y mean/std : {y.mean():.5f} / {y.std():.5f}")

    # 2. Construire et sampler le modèle
    print("\n[2/4] Sampling MCMC (NUTS)…")
    model = build_bayesian_model(X, y, feat_cols)
    trace = run_mcmc(model, draws=draws, tune=tune, chains=chains)

    # 3. Extraire contributions et allocation
    print("\n[3/4] Extraction des contributions et allocation…")
    contributions = extract_asset_contributions(trace, feat_cols)
    allocation    = compute_allocation(contributions, trace, feat_cols)

    # 4. Diagnostics
    print("\n[4/4] Diagnostics MCMC…")
    diag = model_diagnostics(trace, feat_cols)

    # ── Affichage ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALLOCATION RECOMMANDÉE")
    print("=" * 60)
    print(f"\n{'Actif':<14} {'Label':<14} {'Poids':>8} {'IC 5%':>8} {'IC 95%':>8}")
    print("-" * 54)
    for t, row in allocation.iterrows():
        bar = "█" * int(row["weight"] * 40)
        print(f"{t:<14} {row['label']:<14} "
              f"{row['weight']*100:>7.1f}% "
              f"{row['weight_lo']*100:>7.1f}% "
              f"{row['weight_hi']*100:>7.1f}%  {bar}")
    print(f"\n{'Total':<28} {allocation['weight'].sum()*100:>7.1f}%")

    print("\n" + "=" * 60)
    print("  DIAGNOSTICS MCMC")
    print("=" * 60)
    print(f"  R-hat max   : {diag['r_hat_max']:.4f}  {'✓ convergé' if diag['r_hat_max'] < 1.05 else '✗ vérifier'}")
    print(f"  ESS min     : {diag['ess_min']:.0f}    {'✓ ok' if diag['ess_min'] > 200 else '✗ trop faible'}")
    print(f"  N samples   : {diag['n_samples']}")
    if diag["loo_score"]:
        print(f"  LOO score   : {diag['loo_score']:.2f}")

    # ── Sauvegarde ─────────────────────────────────────────────
    if save:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        allocation.to_parquet(out / "allocation.parquet")
        contributions.to_parquet(out / "contributions.parquet")
        trace.to_netcdf(str(out / "mcmc_trace.nc"))

        print(f"\n[OK] Résultats sauvegardés dans {output_dir}/")
        print(f"       allocation.parquet")
        print(f"       contributions.parquet")
        print(f"       mcmc_trace.nc")

    return {
        "allocation":    allocation,
        "contributions": contributions,
        "diagnostics":   diag,
        "trace":         trace,
        "feat_cols":     feat_cols,
        "X_mean":        X_mean,
        "X_std":         X_std,
    }


# ── 7. Prédiction en ligne (inférence sur nouvelles données) ──────────────────

def predict_allocation(
    new_data: pd.DataFrame,
    trace: az.InferenceData,
    feat_cols: list[str],
    X_mean: np.ndarray,
    X_std: np.ndarray,
) -> pd.DataFrame:
    """
    Calcule l'allocation recommandée pour un nouveau snapshot de features.
    Utilisé pour le dashboard temps réel.

    Paramètres
    ----------
    new_data : DataFrame avec les mêmes colonnes que le dataset d'entraînement
               (typiquement : les dernières N lignes du dataset)

    Retourne l'allocation avec IC pour chaque actif.
    """
    available = [c for c in feat_cols if c in new_data.columns]
    X_new = new_data[available].values.astype(float)
    X_new = (X_new - X_mean[:len(available)]) / X_std[:len(available)]

    beta_samples = trace.posterior["beta"].values.reshape(-1, len(feat_cols))
    alpha_samples = trace.posterior["alpha"].values.reshape(-1)

    # Prédiction sur le dernier point (aujourd'hui)
    x_today = X_new[-1]
    all_weights = []

    for i, sample in enumerate(beta_samples):
        b = sample[:len(available)]
        contribs = []
        for t in TICKERS:
            idx = [j for j, c in enumerate(available) if t in c]
            contribs.append(b[idx].sum() if idx else 0.0)
        all_weights.append(softmax(np.array(contribs)))

    all_weights = np.array(all_weights)

    return pd.DataFrame({
        "ticker":    TICKERS,
        "label":     [TICKER_LABELS.get(t, t) for t in TICKERS],
        "weight":    all_weights.mean(axis=0),
        "weight_lo": np.percentile(all_weights, 5, axis=0),
        "weight_hi": np.percentile(all_weights, 95, axis=0),
    }).set_index("ticker")


# ── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.download_yf import fetch_close
    from src.mmm_features import build_mmm_features

    print("=== PortfolioMMM — Pipeline complet ===\n")

    close_wide = fetch_close()
    dataset    = build_mmm_features(close_wide, save=False)
    results    = run_portfolio_mmm(dataset, draws=1000, tune=1000, chains=2)