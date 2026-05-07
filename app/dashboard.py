"""Streamlit Dashboard — Trading Bot Übersicht (Deutsch).

Start:
    streamlit run app/dashboard.py

Tabs:
    1. Heutige Empfehlungen
    2. Trade Journal & Performance
    3. Backtest-Ergebnisse
    4. Postmortems (Fehlertrades)
    5. Markt-Übersicht (Korrelationen, Sektor-Performance)
"""
from __future__ import annotations

import sys
from pathlib import Path

# allow `import config`, `from src...` even if launched from elsewhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from src.model import journal as journal_mod
from src.storage import db


st.set_page_config(
    page_title="Trading Bot — Dashboard",
    page_icon="📈",
    layout="wide",
)


# ----------------------------- Sidebar ---------------------------------

st.sidebar.title("📈 Trading Bot")
st.sidebar.caption("Empfehlungen, Performance, Postmortems")
view = st.sidebar.radio(
    "Ansicht",
    ["Heute", "Trade Journal", "Backtest", "Postmortems", "Markt"],
)

st.sidebar.divider()
summary = journal_mod.journal_summary()
st.sidebar.metric("Predictions gesamt", summary["n_predictions"])
st.sidebar.metric("Davon evaluiert", summary["n_evaluated"])
st.sidebar.metric("Win-Rate", f"{summary['win_rate']*100:.1f}%")
st.sidebar.metric("Ø Realised Return", f"{summary['mean_realised_return']*100:+.2f}%")


# ----------------------------- Helpers ---------------------------------

def _kpi_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, val) in zip(cols, metrics.items()):
        col.metric(label, val)


# ----------------------------- Views -----------------------------------

if view == "Heute":
    st.title("Heutige Empfehlungen")
    st.caption("Vom letzten täglichen Lauf des Orchestrators (`scripts/run_daily.py`).")
    today_path = config.JOURNAL_DIR / "today_recommendations.parquet"
    if not today_path.exists():
        st.info("Noch keine Empfehlungen vorhanden. Erstmal `python scripts/run_daily.py` laufen lassen.")
    else:
        df = pd.read_parquet(today_path)
        long = df[df["action"] == "long"].sort_values("score", ascending=False)
        short = df[df["action"] == "short"].sort_values("score")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 Long-Empfehlungen")
            st.dataframe(long, use_container_width=True, hide_index=True)
        with c2:
            st.subheader("🔴 Short-Empfehlungen")
            st.caption("Im Live-Modus werden Shorts NICHT ausgeführt — nur geloggt zur Validierung.")
            st.dataframe(short, use_container_width=True, hide_index=True)


elif view == "Trade Journal":
    st.title("Trade Journal")
    days = st.slider("Zeitraum (Tage)", 7, 365, 90)
    recent = journal_mod.recent_predictions(days=days)
    if recent.empty:
        st.info("Noch keine Trades im Journal.")
    else:
        st.dataframe(recent, use_container_width=True, hide_index=True)
        st.subheader("Pendend auf Outcome")
        pending = journal_mod.predictions_pending_outcome(horizon_days=5)
        st.dataframe(pending, use_container_width=True, hide_index=True)


elif view == "Backtest":
    st.title("Backtest-Ergebnisse")
    eq_path = config.JOURNAL_DIR / "backtest_equity.parquet"
    tr_path = config.JOURNAL_DIR / "backtest_trades.parquet"
    if not eq_path.exists():
        st.info("Noch kein Backtest gelaufen. `python scripts/run_backtest.py` ausführen.")
    else:
        equity = pd.read_parquet(eq_path)
        trades = pd.read_parquet(tr_path)
        capital_initial = float(equity.iloc[0])
        capital_final = float(equity.iloc[-1])
        total_ret = capital_final / capital_initial - 1
        n_trades = len(trades)
        win_rate = (trades["pnl"] > 0).mean() if n_trades else 0
        _kpi_row({
            "Total Return": f"{total_ret*100:+.1f}%",
            "Anzahl Trades": f"{n_trades}",
            "Win-Rate": f"{win_rate*100:.1f}%",
            "Endkapital": f"{capital_final:,.0f} €",
        })
        st.subheader("Equity Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Kapital"))
        fig.update_layout(yaxis_title="EUR", xaxis_title="Datum", height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Trades")
        st.dataframe(trades.tail(200), use_container_width=True, hide_index=True)


elif view == "Postmortems":
    st.title("Postmortems — Fehler-Analyse")
    failures = journal_mod.failed_trades(limit=50)
    if failures.empty:
        st.info("Keine Fehltrades im Journal (oder alle waren erfolgreich).")
    else:
        for _, row in failures.iterrows():
            with st.expander(
                f"❌ {row['symbol']} {row['action']} am {row['asof_date']} — "
                f"Score {row['score']:+.2f} — Realised {row['realised_return']*100:+.2f}%"
            ):
                st.write(f"**Top-Features:** {row.get('top_features_json', '—')}")
                st.write(f"**Sentiment-Inputs:** {row.get('sentiment_inputs_json', '—')}")
                st.write(f"**Makro:** {row.get('macro_snapshot_json', '—')}")
                # postmortem text if available
                import sqlite3
                with sqlite3.connect(str(journal_mod.DEFAULT_DB)) as c:
                    pm = c.execute(
                        "SELECT analysis_md, llm_model FROM postmortems WHERE prediction_id = ?",
                        (row["id"],),
                    ).fetchone()
                if pm:
                    st.markdown(f"**Analyse ({pm[1]}):**\n\n{pm[0]}")
                else:
                    st.caption("(Noch keine Postmortem-Analyse generiert.)")


elif view == "Markt":
    st.title("Markt-Übersicht")
    try:
        spx = db.read_prices("_GSPC")
        spx_view = spx.tail(252)
        fig = go.Figure(go.Scatter(x=spx_view.index, y=spx_view["adj_close"], name="^GSPC"))
        fig.update_layout(title="S&P 500 — letztes Jahr", height=400)
        st.plotly_chart(fig, use_container_width=True)
    except FileNotFoundError:
        st.warning("Benchmark-Daten fehlen. Notebook 02 laufen lassen.")

    macro_path = config.MACRO_DIR / "fred_panel.parquet"
    if macro_path.exists():
        macro = pd.read_parquet(macro_path).tail(252)
        st.subheader("Makro-Indikatoren (letztes Jahr)")
        for col in ("vix", "fed_funds_rate", "yield_curve_10y_2y"):
            if col in macro.columns:
                fig = go.Figure(go.Scatter(x=macro.index, y=macro[col], name=col))
                fig.update_layout(title=col, height=250, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)
