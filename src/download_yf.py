import yfinance as yf
import pandas as pd

def fetch_close(tickers, start="2000-01-01", end=None):
    df = yf.download(tickers, start=start, end=end, interval="1d",
                     auto_adjust=True, group_by="ticker", progress=False)
    close = pd.concat({t: df[t]["Close"] for t in tickers}, axis=1).sort_index()
    # colonnes propres sans ^
    close.columns = [c.replace("^","") for c in close.columns]
    return close
