# src/download_yf.py

from pathlib import Path
import pandas as pd
import yfinance as yf

# -----------------------------------------------------
# Chemins basés sur l'emplacement de CE fichier
# -----------------------------------------------------

# src/ -> project_root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_TIDY_PATH = RAW_DIR / "ohlcv_10y.parquet"   # fichier racine/data/raw/ohlcv_10y.parquet

TICKER_FCHI = "^FCHI"



# -----------------------------------------------------
# 1) Télécharger un bloc de données et le mettre au
#    format tidy (EXACTEMENT comme dans ton notebook)
# -----------------------------------------------------

def download_block_tidy(tickers, start, end) -> pd.DataFrame:
    """
    Télécharge un bloc OHLCV multi-tickers avec yfinance et
    renvoie un DataFrame au format tidy :

        date | ticker | open | high | low | close | adj_close | volume
    """
    raw = yf.download(
        tickers,
        start=start.date().isoformat(),
        end=end.date().isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,   # on veut les 4 prix bruts + Adj Close séparé
        actions=False,
        progress=False,
        threads=True,
    )

    if raw.empty:
        print(f"[WARN] yf.download a renvoyé vide pour {tickers} entre {start.date()} et {end.date()}")
        return pd.DataFrame(columns=[
            "date", "ticker", "open", "high", "low", "close", "adj_close", "volume"
        ])

    # >>> EXACTEMENT ton code notebook <<<
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

    tidy["date"] = pd.to_datetime(tidy["date"])
    return tidy


# -----------------------------------------------------
# 2) Charger le parquet tidy existant (ou l'initialiser)
# -----------------------------------------------------

def load_or_init_tidy() -> pd.DataFrame:
    """
    Charge data/raw/ohlcv_10y.parquet.
    S'il n'existe pas, télécharge 10 ans pour ^FCHI
    et crée ce fichier.
    """
    if RAW_TIDY_PATH.exists():
        df = pd.read_parquet(RAW_TIDY_PATH)
        df["date"] = pd.to_datetime(df["date"])
        print("[INFO] Tidy existant chargé :", RAW_TIDY_PATH)
        return df

    print("[INFO] Aucun parquet tidy trouvé, initialisation sur 10 ans pour FCHI…")
    end = pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=10)

    tidy = download_block_tidy([TICKER_FCHI], start, end)
    tidy.to_parquet(RAW_TIDY_PATH, index=False)
    print("[OK] Tidy initial créé :", RAW_TIDY_PATH)

    return tidy


# -----------------------------------------------------
# 3) Mettre à jour uniquement FCHI dans le parquet tidy
# -----------------------------------------------------

def update_fchi_tidy() -> pd.DataFrame:
    """
    - Charge le tidy multi-tickers depuis data/raw/ohlcv_10y.parquet
    - Cherche la dernière date disponible pour ^FCHI
    - Télécharge les jours manquants
    - Concatène les nouvelles lignes pour FCHI
    - Ré-écrit data/raw/ohlcv_10y.parquet

    Retourne le tidy complet mis à jour.
    """
    # 1. Charger (ou créer) l'historique tidy
    tidy_all = load_or_init_tidy()

    # 2. Extraire la partie FCHI
    fchi_mask = tidy_all["ticker"] == TICKER_FCHI
    tidy_fchi = tidy_all[fchi_mask].copy()

    if tidy_fchi.empty:
        last_date = None
    else:
        last_date = tidy_fchi["date"].max().normalize()

    print(f"[INFO] Dernière date FCHI dans le parquet tidy : {last_date}")

    today = pd.Timestamp.today().normalize()

    # 3. Déterminer la période à compléter
    if last_date is None:
        # on n'a encore aucune ligne FCHI (cas théorique)
        start = today - pd.DateOffset(years=10)
    else:
        start = last_date + pd.Timedelta(days=1)

    if start > today:
        print("[INFO] Aucune nouvelle date à télécharger pour FCHI (déjà à jour).")
        return tidy_all

    print(f"[INFO] Téléchargement FCHI du {start.date()} au {today.date()}…")

    new_tidy = download_block_tidy([TICKER_FCHI], start, today)

    if new_tidy.empty:
        print("[WARN] Aucunes nouvelles lignes reçues pour FCHI.")
        return tidy_all

    # 4. Mettre à jour uniquement la partie FCHI
    tidy_fchi_updated = (
        pd.concat([tidy_fchi, new_tidy], ignore_index=True)
          .drop_duplicates(subset=["date", "ticker"], keep="last")
          .sort_values("date")
          .reset_index(drop=True)
    )

    # 5. Recomposer le tidy global (autres tickers inchangés)
    tidy_without_fchi = tidy_all[~fchi_mask]
    tidy_all_updated = (
        pd.concat([tidy_without_fchi, tidy_fchi_updated], ignore_index=True)
          .sort_values(["ticker", "date"])
          .reset_index(drop=True)
    )

    # 6. Sauvegarder
    tidy_all_updated.to_parquet(RAW_TIDY_PATH, index=False)
    print("[OK] Parquet tidy mis à jour pour FCHI :", RAW_TIDY_PATH)

    return tidy_all_updated


# -----------------------------------------------------
# 4) Point d’entrée script
# -----------------------------------------------------

if __name__ == "__main__":
    update_fchi_tidy()
