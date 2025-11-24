# app/dashboard.py

from pathlib import Path

import numpy as np
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
def load_features_cac() -> pd.DataFrame:
    df = pd.read_parquet(PROC_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


@st.cache_resource
def load_arima_model() -> ARIMAResults:
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
        - les **prévisions** produites par le modèle ARIMA sauvegardé
        """
    )

    # --------- Chargements communs
    feat_cac = load_features_cac()
    model = load_arima_model()

    feat_cac["date"] = pd.to_datetime(feat_cac["date"])
    feat_cac = feat_cac.sort_values("date")

    close_series = feat_cac.set_index("date")["close"].sort_index()
    logret_series = feat_cac.set_index("date")["logret1"].sort_index()

    # Split temporel 70% / 15% / 15% (comme dans ton notebook)
    unique_dates = feat_cac["date"].sort_values().unique()
    n_dates = len(unique_dates)
    train_end = unique_dates[int(0.70 * n_dates)]
    val_end = unique_dates[int(0.85 * n_dates)]

    # --- Ensemble de test = dates > val_end
    test_s = logret_series[logret_series.index > val_end].dropna()
    test_index = test_s.index

    # 1) Prévisions de logret sur la zone de test
    # (modèle entraîné sur train+val, on fait un forecast multi-step)
    fc_test_log = model.forecast(steps=len(test_s))
    fc_test_log = pd.Series(
        np.asarray(fc_test_log),
        index=test_index,
        name="logret_pred_test",
    )

    # 2) Conversion logret -> prix de clôture prédits sur la zone de test
    last_close_before_test = close_series[close_series.index <= test_index[0]].iloc[-1]

    pred_prices = []
    c = last_close_before_test
    for r in fc_test_log:
        c = c * float(np.exp(r))  # C_t = C_{t-1} * exp(logret_t)
        pred_prices.append(c)

    close_pred_test = pd.Series(pred_prices, index=test_index, name="close_pred")

    # DataFrame pour le 1er graphe : prix réel + prix prédits (test uniquement)
    df_close_plot = pd.concat(
        [
            close_series.rename("close"),
            close_pred_test,
        ],
        axis=1,
    )

    # --------- Layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Historique des prix de clôture (CAC40)")

        st.line_chart(df_close_plot, use_container_width=True)
        st.caption(
            "La courbe **close** correspond au prix réel.\n"
            "La courbe **close_pred** correspond au prix prédit par ARIMA "
            "sur la période de test (~15% des dernières dates)."
        )

    with col_right:
        st.subheader("Paramètres de prévision future")

        horizon = st.slider(
            "Horizon de prévision (jours)",
            min_value=1,
            max_value=10,
            value=5,
        )
        st.caption(
            "Les prévisions futures sont faites sur les **log-returns** "
            "puis converties en prix."
        )

    # --------- 3) Log-returns réel vs prédit sur la zone de test
    st.subheader("Log-returns : réel vs prédit – Ensemble de test")

    test_logret_real = test_s.rename("logret_test")
    df_logret_plot = pd.concat([test_logret_real, fc_test_log], axis=1)

    st.line_chart(df_logret_plot, use_container_width=True)

    # --------- 4) Prévisions futures (au-delà de la dernière date)
    st.subheader("Prévisions futures (prix)")

    # Prévisions futures de logret (multi-step après la fin de la série d'entraînement)
    fc_future_log = model.forecast(steps=horizon)
    future_index = pd.date_range(
        close_series.index[-1], periods=horizon + 1, freq="B"
    )[1:]
    fc_future_log = pd.Series(
        np.asarray(fc_future_log),
        index=future_index,
        name="logret_future",
    )

    # Conversion en prix futurs
    last_close = close_series.iloc[-1]
    future_prices = []
    c = last_close
    for r in fc_future_log:
        c = c * float(np.exp(r))
        future_prices.append(c)

    fc_future_close = pd.Series(
        future_prices, index=future_index, name="close_future_pred"
    )

    hist_future = close_series.tail(100).rename("close")
    df_future_plot = pd.concat([hist_future, fc_future_close], axis=1)

    st.line_chart(df_future_plot, use_container_width=True)

    st.write("Dernières prévisions futures (prix) :")
    st.dataframe(fc_future_close.to_frame())


if __name__ == "__main__":
    main()
