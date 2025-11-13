import pathlib as p
from datetime import datetime
import pandas as pd

from src.config import TICKERS, START_DATE, HORIZON_D, FEATURE_LAGS, FEATURE_WINDOWS, RAW_DIR, PROC_DIR
from src.download_yf import fetch_close
from src.features import make_dataset

def main():
    p.Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
    p.Path(PROC_DIR).mkdir(parents=True, exist_ok=True)

    close = fetch_close(TICKERS, start=START_DATE)
    # snapshot brut
    snap_name = f"{RAW_DIR}/close_full.parquet"
    close.to_parquet(snap_name)

    ds = make_dataset(close, FEATURE_WINDOWS, FEATURE_LAGS, HORIZON_D)
    out = f"{PROC_DIR}/dataset_h{HORIZON_D}d.parquet"
    ds.to_parquet(out, index=False)

    # petit résumé pour log
    print("Snapshot brut:", snap_name, "| lignes:", len(close))
    print("Dataset :", out, "| lignes:", len(ds), "| colonnes:", list(ds.columns))

if __name__ == "__main__":
    main()
