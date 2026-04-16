# app/streamlit_app.py
# PortfolioMMM — Dashboard temps réel
# ====================================================
# Lancement : streamlit run app/streamlit_app.py
# ====================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys, os

# ── Imports projet (robuste si lancé depuis racine ou depuis app/) ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from src.download_yf      import fetch_close, update
    from src.mmm_features     import build_mmm_features, get_feature_groups
    from src.bayes_allocation import run_portfolio_mmm, predict_allocation
    from src.config           import TICKERS, TICKER_LABELS, ADSTOCK_HALFLIFE, SATURATION_SCALE
    LIVE_MODE = True
except ImportError:
    LIVE_MODE = False   # mode démo avec données simulées


# ── Config page ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioMMM",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette & style ──────────────────────────────────────────────────
COLORS = {
    "GC=F":     "#F5C842",
    "CL=F":     "#4E9AF1",
    "EURUSD=X": "#6BCB77",
    "BTC-USD":  "#FF7043",
    "bg":       "#0D0F14",
    "surface":  "#161920",
    "border":   "#252A35",
    "text":     "#E8EAF0",
    "muted":    "#6B7280",
}

LABELS = {
    "GC=F":     "Or",
    "CL=F":     "Pétrole WTI",
    "EURUSD=X": "EUR/USD",
    "BTC-USD":  "Bitcoin",
}

TICKERS_LIST = ["GC=F", "CL=F", "EURUSD=X", "BTC-USD"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0D0F14;
    color: #E8EAF0;
}
.stApp { background-color: #0D0F14; }

/* Header principal */
.mmm-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: #6B7280;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.mmm-title {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #E8EAF0;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.mmm-subtitle {
    font-size: 14px;
    color: #6B7280;
    margin-top: 6px;
}

/* Metric cards */
.metric-card {
    background: #161920;
    border: 1px solid #252A35;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #6B7280;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 26px;
    font-weight: 700;
    color: #E8EAF0;
    line-height: 1;
}
.metric-sub {
    font-size: 12px;
    color: #6B7280;
    margin-top: 4px;
}

/* Allocation bars */
.alloc-row {
    display: flex;
    align-items: center;
    margin-bottom: 14px;
    gap: 12px;
}
.alloc-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: #E8EAF0;
    width: 80px;
    flex-shrink: 0;
}
.alloc-bar-track {
    flex: 1;
    height: 8px;
    background: #252A35;
    border-radius: 4px;
    position: relative;
    overflow: visible;
}
.alloc-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.alloc-ic {
    position: absolute;
    height: 100%;
    top: 0;
    border-radius: 4px;
    opacity: 0.25;
}
.alloc-pct {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: #E8EAF0;
    width: 44px;
    text-align: right;
    flex-shrink: 0;
}
.alloc-ic-text {
    font-size: 10px;
    color: #6B7280;
    width: 90px;
    flex-shrink: 0;
    text-align: right;
}

/* Section titles */
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.2em;
    color: #6B7280;
    text-transform: uppercase;
    border-bottom: 1px solid #252A35;
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 28px;
}

/* Status badge */
.badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.1em;
    padding: 3px 8px;
    border-radius: 4px;
    text-transform: uppercase;
}
.badge-live { background: #0F2918; color: #6BCB77; border: 1px solid #1a4a2a; }
.badge-demo { background: #1a1506; color: #F5C842; border: 1px solid #3a2e0a; }

/* MMM insight box */
.insight-box {
    background: #161920;
    border-left: 3px solid #4E9AF1;
    border-radius: 0 8px 8px 0;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.insight-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.1em;
    color: #4E9AF1;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.insight-text {
    font-size: 13px;
    color: #B0B7C3;
    line-height: 1.6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0D0F14;
    border-right: 1px solid #252A35;
}
</style>
""", unsafe_allow_html=True)


# ── Données ──────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    """Charge ou simule les données du portefeuille."""
    if LIVE_MODE:
        try:
            close_wide   = fetch_close()
            dataset      = build_mmm_features(close_wide, save=False)
            results      = run_portfolio_mmm(dataset, draws=500, tune=500,
                                             chains=2, save=False)
            allocation   = results["allocation"]
            feat_groups  = get_feature_groups(dataset)
            return close_wide, dataset, allocation, feat_groups, True
        except Exception as e:
            st.warning(f"Mode live indisponible ({e}). Données simulées.")

    # ── Simulation (démo) ─────────────────────────────────────────
    np.random.seed(42)
    dates  = pd.date_range("2023-01-01", "2025-04-15", freq="B")
    starts = {"GC=F": 1800, "CL=F": 75, "EURUSD=X": 1.07, "BTC-USD": 16500}
    vols   = {"GC=F": 0.008, "CL=F": 0.025, "EURUSD=X": 0.004, "BTC-USD": 0.04}

    frames = []
    for t in TICKERS_LIST:
        r = np.random.normal(0.0003, vols[t], len(dates))
        p = starts[t] * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({"date": dates, "ticker": t, "close": p}))

    close_wide = pd.concat(frames).pivot(
        index="date", columns="ticker", values="close"
    )

    # Features MMM simulées
    rets = close_wide.pct_change()
    ADSTOCK_HL = {"GC=F": 10, "CL=F": 5, "EURUSD=X": 7, "BTC-USD": 3}
    SAT_SCALE  = {"GC=F": 0.05, "CL=F": 0.10, "EURUSD=X": 0.03, "BTC-USD": 0.20}

    dataset = close_wide.copy()
    for t in TICKERS_LIST:
        r = rets[t].fillna(0)
        alpha_ads = 1 - np.exp(-np.log(2) / ADSTOCK_HL[t])
        dataset[f"adstock_{t}"] = r.ewm(alpha=alpha_ads, adjust=False).mean()
        cum = r.rolling(20).sum().fillna(0)
        dataset[f"sat_{t}"] = 2 / (1 + np.exp(-cum / SAT_SCALE[t])) - 1

    # Allocation simulée avec légère asymétrie BTC > Or
    allocation = pd.DataFrame({
        "ticker":    TICKERS_LIST,
        "label":     [LABELS[t] for t in TICKERS_LIST],
        "weight":    [0.293, 0.207, 0.191, 0.309],
        "weight_lo": [0.262, 0.165, 0.166, 0.287],
        "weight_hi": [0.342, 0.245, 0.216, 0.355],
        "contrib_mean": [0.499, 0.150, 0.070, 0.550],
    }).set_index("ticker")

    feat_groups = {
        "adstock":    [f"adstock_{t}" for t in TICKERS_LIST],
        "saturation": [f"sat_{t}" for t in TICKERS_LIST],
    }
    return close_wide, dataset, allocation, feat_groups, False


def compute_portfolio_perf(close_wide, allocation):
    """Calcule la performance du portefeuille pondéré vs equal-weight."""
    rets = close_wide.pct_change().dropna()
    weights = {t: allocation.loc[t, "weight"] for t in TICKERS_LIST if t in allocation.index}

    port_ret_mmm = sum(
        weights.get(t, 0.25) * rets[t] for t in TICKERS_LIST if t in rets.columns
    )
    port_ret_eq = rets[TICKERS_LIST].mean(axis=1)

    cum_mmm = (1 + port_ret_mmm).cumprod()
    cum_eq  = (1 + port_ret_eq).cumprod()

    sharpe   = port_ret_mmm.mean() / (port_ret_mmm.std() + 1e-9) * np.sqrt(252)
    max_dd   = (cum_mmm / cum_mmm.cummax() - 1).min()
    ytd_ret  = cum_mmm.iloc[-1] / cum_mmm[cum_mmm.index >= f"{cum_mmm.index[-1].year}-01-01"].iloc[0] - 1

    return {
        "cum_mmm":  cum_mmm,
        "cum_eq":   cum_eq,
        "dates":    rets.index,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
        "ytd_ret":  ytd_ret,
        "port_ret": port_ret_mmm,
    }


# ── Sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:24px">
        <div class="mmm-header">PortfolioMMM</div>
        <div style="font-family:'Space Mono',monospace;font-size:20px;
                    font-weight:700;color:#E8EAF0;">Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Modèle</div>', unsafe_allow_html=True)

    horizon = st.selectbox(
        "Horizon de prédiction",
        options=[1, 5, 10],
        format_func=lambda x: f"J+{x}",
        index=0,
    )

    retrain = st.button("↺  Ré-entraîner le modèle", use_container_width=True)
    if retrain:
        st.cache_data.clear()
        st.rerun()

    st.markdown('<div class="section-title">Affichage</div>', unsafe_allow_html=True)

    show_ic     = st.toggle("Intervalles de confiance", value=True)
    show_mmm    = st.toggle("Features MMM (adstock/sat)", value=True)
    lookback    = st.slider("Historique (jours)", 60, 500, 252)

    st.markdown('<div class="section-title">À propos</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:12px;color:#6B7280;line-height:1.7">
        Chaque actif = un "canal média"<br>
        Rendement portfolio = "les ventes"<br>
        <br>
        <b style="color:#B0B7C3">Adstock</b> → mémoire du momentum<br>
        <b style="color:#B0B7C3">Saturation</b> → effet décroissant<br>
        <b style="color:#B0B7C3">Bayes</b> → incertitude sur les poids
    </div>
    """, unsafe_allow_html=True)


# ── Chargement ───────────────────────────────────────────────────────

with st.spinner("Chargement des données…"):
    close_wide, dataset, allocation, feat_groups, is_live = load_data()

perf = compute_portfolio_perf(close_wide, allocation)


# ── HEADER ───────────────────────────────────────────────────────────

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown("""
    <div class="mmm-header">Marketing Mix Model appliqué à la finance</div>
    <div class="mmm-title">PortfolioMMM</div>
    <div class="mmm-subtitle">
        Allocation bayésienne · Or · Pétrole · EUR/USD · Bitcoin
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    badge_class = "badge-live" if is_live else "badge-demo"
    badge_text  = "Live" if is_live else "Démo"
    st.markdown(
        f'<div style="text-align:right;margin-top:16px">'
        f'<span class="badge {badge_class}">{badge_text}</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)


# ── MÉTRIQUES CLÉS ───────────────────────────────────────────────────

st.markdown('<div class="section-title">Métriques du portefeuille</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sharpe ratio</div>
        <div class="metric-value">{perf['sharpe']:.2f}</div>
        <div class="metric-sub">annualisé (252j)</div>
    </div>""", unsafe_allow_html=True)

with m2:
    dd_color = "#FF7043" if perf['max_dd'] < -0.15 else "#6BCB77"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max drawdown</div>
        <div class="metric-value" style="color:{dd_color}">{perf['max_dd']*100:.1f}%</div>
        <div class="metric-sub">depuis le lancement</div>
    </div>""", unsafe_allow_html=True)

with m3:
    ytd_color = "#6BCB77" if perf['ytd_ret'] > 0 else "#FF7043"
    ytd_sign  = "+" if perf['ytd_ret'] > 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rendement YTD</div>
        <div class="metric-value" style="color:{ytd_color}">{ytd_sign}{perf['ytd_ret']*100:.1f}%</div>
        <div class="metric-sub">depuis le 1er janvier</div>
    </div>""", unsafe_allow_html=True)

with m4:
    last_ret = perf["port_ret"].iloc[-1]
    ret_color = "#6BCB77" if last_ret > 0 else "#FF7043"
    ret_sign  = "+" if last_ret > 0 else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Dernier rendement</div>
        <div class="metric-value" style="color:{ret_color}">{ret_sign}{last_ret*100:.2f}%</div>
        <div class="metric-sub">{perf['dates'][-1].strftime('%d %b %Y')}</div>
    </div>""", unsafe_allow_html=True)


# ── ALLOCATION + PERFORMANCE (colonnes principales) ──────────────────

col_alloc, col_perf = st.columns([1, 2], gap="large")

with col_alloc:
    st.markdown('<div class="section-title">Allocation recommandée</div>',
                unsafe_allow_html=True)

    alloc_html = ""
    for t in TICKERS_LIST:
        if t not in allocation.index:
            continue
        row   = allocation.loc[t]
        w     = row["weight"] * 100
        lo    = row["weight_lo"] * 100
        hi    = row["weight_hi"] * 100
        color = COLORS[t]
        label = LABELS[t]

        ic_bar = (
            f'<div class="alloc-ic" style="left:{lo:.1f}%;width:{hi-lo:.1f}%;'
            f'background:{color}"></div>'
        ) if show_ic else ""

        ic_text = (
            f'<div class="alloc-ic-text">{lo:.0f}–{hi:.0f}%</div>'
        ) if show_ic else ""

        alloc_html += f"""
        <div class="alloc-row">
            <div class="alloc-label">{label}</div>
            <div class="alloc-bar-track">
                {ic_bar}
                <div class="alloc-bar-fill" style="width:{w:.1f}%;background:{color}"></div>
            </div>
            <div class="alloc-pct">{w:.1f}%</div>
            {ic_text}
        </div>"""

    st.markdown(alloc_html, unsafe_allow_html=True)

    # Donut chart
    fig_donut = go.Figure(go.Pie(
        labels=[LABELS[t] for t in TICKERS_LIST if t in allocation.index],
        values=[allocation.loc[t, "weight"] * 100 for t in TICKERS_LIST if t in allocation.index],
        hole=0.65,
        marker_colors=[COLORS[t] for t in TICKERS_LIST if t in allocation.index],
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{sum(allocation['weight']*100):.0f}%</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color="#E8EAF0", family="Space Mono"),
    )
    fig_donut.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_donut, use_container_width=True)


with col_perf:
    st.markdown('<div class="section-title">Performance cumulée</div>',
                unsafe_allow_html=True)

    dates_slice = perf["dates"][-lookback:]
    cum_mmm     = perf["cum_mmm"].reindex(dates_slice)
    cum_eq      = perf["cum_eq"].reindex(dates_slice)

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=dates_slice, y=cum_mmm,
        name="PortfolioMMM",
        line=dict(color="#4E9AF1", width=2),
        fill="tozeroy",
        fillcolor="rgba(78,154,241,0.06)",
        hovertemplate="%{y:.3f}<extra>MMM</extra>",
    ))
    fig_perf.add_trace(go.Scatter(
        x=dates_slice, y=cum_eq,
        name="Equal-weight",
        line=dict(color="#6B7280", width=1.5, dash="dot"),
        hovertemplate="%{y:.3f}<extra>EW</extra>",
    ))
    fig_perf.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=8, b=8, l=0, r=0),
        height=220,
        legend=dict(
            font=dict(color="#6B7280", size=11),
            bgcolor="rgba(0,0,0,0)",
            x=0, y=1,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(color="#6B7280", size=10),
            tickcolor="#252A35",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1E222E",
            zeroline=False,
            tickfont=dict(color="#6B7280", size=10),
            tickformat=".2f",
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # Rendements quotidiens
    st.markdown('<div class="section-title">Rendements quotidiens</div>',
                unsafe_allow_html=True)

    daily = perf["port_ret"].reindex(dates_slice)
    colors_bar = ["#6BCB77" if r >= 0 else "#FF7043" for r in daily]

    fig_daily = go.Figure(go.Bar(
        x=dates_slice, y=daily * 100,
        marker_color=colors_bar,
        hovertemplate="%{y:.2f}%<extra></extra>",
    ))
    fig_daily.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=120,
        xaxis=dict(showgrid=False, tickfont=dict(color="#6B7280", size=9)),
        yaxis=dict(
            showgrid=True, gridcolor="#1E222E",
            tickfont=dict(color="#6B7280", size=9),
            ticksuffix="%",
        ),
        bargap=0.15,
    )
    st.plotly_chart(fig_daily, use_container_width=True)


# ── FEATURES MMM ─────────────────────────────────────────────────────

if show_mmm:
    st.markdown('<div class="section-title">Features MMM — Adstock & Saturation</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Adstock (mémoire du momentum)", "Saturation (effet décroissant)"])

    rets = close_wide.pct_change().dropna()

    with tab1:
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Analogie MMM</div>
            <div class="insight-text">
                L'adstock mesure la persistance d'un choc de rendement dans le temps.
                Un adstock élevé = momentum positif qui n'a pas encore été "digéré" par le marché.
                Demi-vies : Or 10j · Pétrole 5j · EUR/USD 7j · BTC 3j
            </div>
        </div>""", unsafe_allow_html=True)

        fig_ads = go.Figure()
        ADSTOCK_HL = {"GC=F": 10, "CL=F": 5, "EURUSD=X": 7, "BTC-USD": 3}
        for t in TICKERS_LIST:
            if t not in rets.columns:
                continue
            r     = rets[t].fillna(0)
            alpha = 1 - np.exp(-np.log(2) / ADSTOCK_HL[t])
            ads   = r.ewm(alpha=alpha, adjust=False).mean()
            slice_ads = ads[-lookback:]
            fig_ads.add_trace(go.Scatter(
                x=slice_ads.index,
                y=slice_ads.values * 100,
                name=LABELS[t],
                line=dict(color=COLORS[t], width=1.5),
                hovertemplate=f"{LABELS[t]}: %{{y:.3f}}%<extra></extra>",
            ))
        fig_ads.add_hline(y=0, line_color="#252A35", line_width=1)
        fig_ads.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            legend=dict(font=dict(color="#6B7280", size=11), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#6B7280", size=10)),
            yaxis=dict(showgrid=True, gridcolor="#1E222E",
                       tickfont=dict(color="#6B7280", size=10), ticksuffix="%"),
        )
        st.plotly_chart(fig_ads, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="insight-box">
            <div class="insight-title">Analogie MMM</div>
            <div class="insight-text">
                La saturation (sigmoïde des rendements cumulés 20j) détecte quand un actif
                est suracheté (+1) ou survendu (-1). Un actif saturé à +0.9 apporte peu
                de rendement marginal supplémentaire — on sous-pondère.
            </div>
        </div>""", unsafe_allow_html=True)

        fig_sat = go.Figure()
        SAT_SCALE = {"GC=F": 0.05, "CL=F": 0.10, "EURUSD=X": 0.03, "BTC-USD": 0.20}
        for t in TICKERS_LIST:
            if t not in rets.columns:
                continue
            r   = rets[t].fillna(0)
            cum = r.rolling(20).sum().fillna(0)
            sat = 2 / (1 + np.exp(-cum / SAT_SCALE[t])) - 1
            slice_sat = sat[-lookback:]
            fig_sat.add_trace(go.Scatter(
                x=slice_sat.index,
                y=slice_sat.values,
                name=LABELS[t],
                line=dict(color=COLORS[t], width=1.5),
                hovertemplate=f"{LABELS[t]}: %{{y:.3f}}<extra></extra>",
            ))
        fig_sat.add_hline(y=0,    line_color="#252A35", line_width=1)
        fig_sat.add_hline(y=0.7,  line_color="#FF7043", line_width=0.8, line_dash="dot")
        fig_sat.add_hline(y=-0.7, line_color="#6BCB77", line_width=0.8, line_dash="dot")
        fig_sat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=0, b=0, l=0, r=0),
            height=200,
            legend=dict(font=dict(color="#6B7280", size=11), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(showgrid=False, tickfont=dict(color="#6B7280", size=10)),
            yaxis=dict(
                showgrid=True, gridcolor="#1E222E",
                tickfont=dict(color="#6B7280", size=10),
                range=[-1.1, 1.1],
            ),
        )
        st.plotly_chart(fig_sat, use_container_width=True)


# ── PRIX DES ACTIFS ───────────────────────────────────────────────────

st.markdown('<div class="section-title">Prix des actifs</div>', unsafe_allow_html=True)

price_cols = st.columns(4)
for i, t in enumerate(TICKERS_LIST):
    if t not in close_wide.columns:
        continue
    prices      = close_wide[t].dropna()
    last_price  = prices.iloc[-1]
    prev_price  = prices.iloc[-2]
    chg         = (last_price / prev_price - 1) * 100
    chg_color   = "#6BCB77" if chg >= 0 else "#FF7043"
    chg_sign    = "+" if chg >= 0 else ""
    sparkline_y = prices[-60:].values
    sparkline_x = list(range(len(sparkline_y)))
    spark_color = COLORS[t]

    fig_spark = go.Figure(go.Scatter(
        x=sparkline_x, y=sparkline_y,
        line=dict(color=spark_color, width=1.5),
        fill="tozeroy",
        fillcolor=f"rgba({int(spark_color[1:3],16)},{int(spark_color[3:5],16)},{int(spark_color[5:7],16)},0.08)",
    ))
    fig_spark.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=0),
        height=55,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    with price_cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{LABELS[t]}</div>
            <div style="display:flex;align-items:baseline;gap:8px">
                <span class="metric-value" style="font-size:20px">{last_price:,.2f}</span>
                <span style="font-family:'Space Mono',monospace;font-size:12px;
                             color:{chg_color}">{chg_sign}{chg:.2f}%</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(fig_spark, use_container_width=True)


# ── FOOTER ────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-top:1px solid #252A35;margin-top:40px;padding-top:16px;
            display:flex;justify-content:space-between;align-items:center">
    <div style="font-family:'Space Mono',monospace;font-size:10px;
                color:#3A3F4E;letter-spacing:0.1em">
        PORTFOLIOMMM · RÉGRESSION BAYÉSIENNE · ADSTOCK · SATURATION
    </div>
    <div style="font-size:11px;color:#3A3F4E">
        Données : Yahoo Finance · Modèle : PyMC 5
    </div>
</div>
""", unsafe_allow_html=True)