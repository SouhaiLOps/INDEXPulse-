# =============================================================
# PortfolioMMM — Configuration
# =============================================================
# Chaque actif est un "canal média" dont on veut estimer
# la contribution au rendement du portefeuille.
 
# ── Actifs ────────────────────────────────────────────────────
TICKERS = ["GC=F", "CL=F", "EURUSD=X", "BTC-USD"]
 
TICKER_LABELS = {
    "GC=F":      "Or",
    "CL=F":      "Pétrole WTI",
    "EURUSD=X":  "EUR/USD",
    "BTC-USD":   "Bitcoin",
}
 
# Proxies ETF pour fallback si yfinance rate le futures
ETF_PROXIES = {
    "GC=F":     "GLD",
    "CL=F":     "USO",
    "EURUSD=X": "FXE",
    "BTC-USD":  "IBIT",
}
 
# ── Horizon & features ────────────────────────────────────────
START_DATE    = "2018-01-01"   # BTC disponible proprement depuis 2017
HORIZON_D     = 1              # rendement cible à J+1 (tester aussi 5, 10)
 
FEATURE_WINDOWS = [5, 20, 60]  # courts, moyens, longs
FEATURE_LAGS    = [1, 2, 5]
 
# ── Adstock ───────────────────────────────────────────────────
# Demi-vie de la mémoire du momentum (en jours)
# Analogue MMM : combien de jours un choc de rendement persiste
ADSTOCK_HALFLIFE = {
    "GC=F":     10,   # Or : mémoire longue (actif refuge)
    "CL=F":     5,    # Pétrole : mémoire courte (volatile)
    "EURUSD=X": 7,    # Forex : intermédiaire
    "BTC-USD":  3,    # BTC : mémoire très courte (réactif)
}
 
# ── Saturation ────────────────────────────────────────────────
# Seuil de saturation (rendement cumulé au-delà duquel l'effet
# marginal diminue) — analogue MMM : point d'inflexion sigmoïde
SATURATION_SCALE = {
    "GC=F":     0.05,   # ±5% = zone de saturation Or
    "CL=F":     0.10,   # ±10% Pétrole
    "EURUSD=X": 0.03,   # ±3% Forex (peu volatile)
    "BTC-USD":  0.20,   # ±20% BTC
}
 
# ── Dossiers ──────────────────────────────────────────────────
RAW_DIR  = "data/raw"
PROC_DIR = "data/processed"
 