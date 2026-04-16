# src/download_yf.py
# PortfolioMMM — téléchargement et mise à jour des 4 actifs
# =============================================================
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.config import (
    TICKERS, ETF_PROXIES, START_DATE,
    RAW_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH     = PROJECT_ROOT / RAW_DIR / "ohlcv_portfolio.parquet"
Path(PROJECT_ROOT / RAW_DIR).mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────

def _to_tidy(raw: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Convertit le DataFrame multi-level de yfinance en format tidy :
        date | ticker | close | volume
    """
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])

    # yfinance retourne (metric, ticker) en colonnes multi-level
    frames = []
    for t in tickers:
        try:
            sub = pd.DataFrame({
                "date":   raw.index,
                "ticker": t,
                "close":  raw["Close"][t].values   if "Close"  in raw.columns.get_level_values(0) else raw[t]["Close"].values,
                "volume": raw["Volume"][t].values  if "Volume" in raw.columns.get_level_values(0) else raw[t]["Volume"].values,
            })
            frames.append(sub)
        except Exception as e:
            print(f"[WARN] {t} ignoré : {e}")

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["close"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def _download(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les tickers. Si un ticker échoue, tente le proxy ETF.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
        group_by="column",    # (metric, ticker)
    )
    tidy = _to_tidy(raw, tickers)

    # Fallback ETF pour les tickers vides
    missing = [t for t in tickers if t not in tidy["ticker"].unique()]
    for t in missing:
        proxy = ETF_PROXIES.get(t)
        if proxy:
            print(f"[INFO] {t} vide — tentative proxy {proxy}")
            raw2 = yf.download(proxy, start=start, end=end, interval="1d",
                               auto_adjust=False, progress=False)
            if not raw2.empty:
                sub = pd.DataFrame({
                    "date":   raw2.index,
                    "ticker": t,            # on garde le nom original
                    "close":  raw2["Close"].values,
                    "volume": raw2["Volume"].values,
                })
                sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
                tidy = pd.concat([tidy, sub], ignore_index=True)
                print(f"[OK] {t} récupéré via proxy {proxy}")

    return tidy


# ── Fonctions publiques ───────────────────────────────────────

def load_or_init() -> pd.DataFrame:
    """
    Charge le parquet existant ou initialise depuis START_DATE.
    """
    if RAW_PATH.exists():
        df = pd.read_parquet(RAW_PATH)
        df["date"] = pd.to_datetime(df["date"])
        print(f"[INFO] Parquet chargé : {RAW_PATH} ({len(df):,} lignes)")
        return df

    print(f"[INFO] Initialisation depuis {START_DATE}…")
    end = pd.Timestamp.today().normalize()
    tidy = _download(TICKERS, start=START_DATE, end=end.date().isoformat())
    tidy.to_parquet(RAW_PATH, index=False)
    print(f"[OK] Parquet créé : {len(tidy):,} lignes → {RAW_PATH}")
    return tidy


def update() -> pd.DataFrame:
    """
    Télécharge uniquement les jours manquants pour chaque ticker.
    Ré-écrit le parquet consolidé.
    """
    tidy_all = load_or_init()
    today    = pd.Timestamp.today().normalize()
    new_rows = []

    for t in TICKERS:
        sub      = tidy_all[tidy_all["ticker"] == t]
        last     = sub["date"].max() if not sub.empty else pd.Timestamp(START_DATE)
        start    = (last + pd.Timedelta(days=1)).date().isoformat()

        if pd.Timestamp(start) > today:
            print(f"[INFO] {t} déjà à jour ({last.date()})")
            continue

        print(f"[INFO] {t} : téléchargement {start} → {today.date()}")
        chunk = _download([t], start=start, end=today.date().isoformat())
        if not chunk.empty:
            new_rows.append(chunk)
            print(f"[OK]   {t} : +{len(chunk)} lignes")
        else:
            print(f"[WARN] {t} : aucune donnée reçue")

    if new_rows:
        tidy_all = (
            pd.concat([tidy_all] + new_rows, ignore_index=True)
              .drop_duplicates(subset=["date", "ticker"], keep="last")
              .sort_values(["ticker", "date"])
              .reset_index(drop=True)
        )
        tidy_all.to_parquet(RAW_PATH, index=False)
        print(f"[OK] Parquet mis à jour : {len(tidy_all):,} lignes totales")

    return tidy_all


def fetch_close(tickers: list = None, start: str = START_DATE) -> pd.DataFrame:
    """
    Interface simple pour build_dataset.py :
    Retourne un DataFrame wide  date × ticker  avec les prix de clôture.
    """
    tickers = tickers or TICKERS
    tidy    = load_or_init()
    tidy    = tidy[tidy["ticker"].isin(tickers)]

    close_wide = (
        tidy.pivot(index="date", columns="ticker", values="close")
            .sort_index()
    )
    close_wide = close_wide[tidy["date"] >= start] if start else close_wide
    return close_wide


# ── Point d'entrée ────────────────────────────────────────────
if __name__ == "__main__":
    df = update()
    print("\n=== Résumé ===")
    print(df.groupby("ticker").agg(
        debut=("date", "min"),
        fin=("date", "max"),
        lignes=("date", "count"),
        close_last=("close", "last"),
    ).to_string())