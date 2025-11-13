# Liste des indices (Yahoo) + proxies ETF si besoin
TICKERS = ["^GSPC","^DJI","^IXIC","^FCHI","^GDAXI","^FTSE","^STOXX50E","^N225","^HSI","^KS11"]
ETF_PROXIES = {"^GSPC":"SPY","^DJI":"DIA","^IXIC":"QQQ","^FCHI":"EWQ","^GDAXI":"EWG","^FTSE":"EWU"}

START_DATE = "2000-01-01"
HORIZON_D = 1           # prédire le rendement à J+1 (tu pourras tester 5, 10…)
FEATURE_WINDOWS = [5,20,60]
FEATURE_LAGS = [1,2,5]
RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
