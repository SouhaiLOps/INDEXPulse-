# app/dashboard.py

from pathlib import Path

import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMAResults

# -------------------------------------------------
# Chemins basés sur la racine du projet
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ohlcv_10y.parquet"
PROC_PATH = PROJECT_ROOT / "data" / "processed" / "FCHI_features_h1d.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "arima_logret1_FCHI.pkl"

TICKER_CAC40 = "^FCHI"


# -------------------------------------------------
# Fonctions de chargement (cachées pour performance)
# -------------------------------------------------
@st.cache_data
def load_raw_cac():
    df = pd.read_parquet(RAW_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"] == TICKER_CAC40].sort_values("date")
    return df


@st.cache_data
def load_features_cac():
    df = pd.read_parquet(PROC_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


@st.cache_resource
def load_arima_model():
    return ARIMAResults.load(MODEL_PATH)


# -------------------------------------------------
# App Streamlit
# -------------------------------------------------
def main():
    st.set_page_config(page_title="IndexPulse – CAC40 ARIMA", layout="wide")

    st.title("📈 IndexPulse – CAC40")
    st.markdown(
        """
        Petit dashboard pour suivre :
        - l'historique du **CAC40**  
        - les **prévisions de log-return** produites par le modèle ARIMA sauvegardé
        """
    )

    # --- Colonnes pour mise en page
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Historique des prix de clôture (CAC40)")

        raw_cac = load_raw_cac()
        st.line_chart(
            raw_cac.set_index("date")["close"],
            use_container_width=True,
        )

    with col_right:
        st.subheader("Paramètres de prévision")

        horizon = st.slider("Horizon de prévision (jours)", min_value=1, max_value=10, value=5)

        st.caption("Les prévisions sont faites sur la série **logret1** (rendements log).")

    # --- Zone prédiction ARIMA
    st.subheader("Prévisions ARIMA sur les log-returns")

    feat_cac = load_features_cac()
    series = feat_cac.set_index("date")["logret1"].sort_index()

    model = load_arima_model()

    # Prévision pour N jours
    fc = model.forecast(steps=horizon)

    # On construit un index de dates business à partir de la dernière date observée
    last_date = series.index[-1]
    future_index = pd.date_range(last_date, periods=horizon + 1, freq="B")[1:]  # on saute last_date
    fc = pd.Series(fc, index=future_index, name="forecast_logret1")

    # DataFrame pour visualiser dernier historique + prévisions
    hist_window = 100  # nb de jours d'historique à afficher
    hist = series.tail(hist_window).rename("logret1")

    df_plot = pd.concat([hist, fc], axis=0)

    st.line_chart(df_plot, use_container_width=True)

    st.write("Dernières prévisions :")
    st.dataframe(fc.to_frame())


if __name__ == "__main__":
    main()
