"""Cross-stock, sector, and market-wide sentiment features.

The base sentiment module aggregates news *per ticker per day*. That's
necessary but not sufficient — markets are coupled. Apple's earnings
miss prints a -2% on Apple, but it also moves Skyworks (supplier),
the entire QQQ, and risk-on plays globally.

This module adds four feature families on top of per-stock sentiment:

  1. **Market sentiment** — aggregate over RSS market news (Reuters,
     CNBC, Fed, …). Captures macro/regime news that affect everything.
  2. **Sector sentiment** — mean sentiment over all stocks in the same
     GICS sector. Sector rotation, anyone?
  3. **Peer sentiment** — correlation-weighted sentiment of the most
     correlated peers (excluding self). The model can then learn
     "when my biggest co-mover gets bad news, I usually drop too".
  4. **Sentiment lags + momentum** — t-1, t-3, t-7 history, plus
     the trend (today minus 7 days). News effects often persist or
     compound over multiple days.

The module is **batched** — it computes all features for a panel of
symbols × dates in one pass, so adding it to `build_dataset` doesn't
explode runtime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import NEWS_DIR
from src.data import universe


# --------------------------- per-symbol panel ---------------------------

def _per_symbol_sentiment_panel(symbols: list[str]) -> pd.DataFrame:
    """Wide DataFrame: (date, symbol) -> sent_mean, sent_max_neg, n_articles.

    Reads the same per-symbol news files used in build_dataset, but
    pivots them to long format so we can aggregate cross-stock cheaply.
    """
    rows = []
    for sym in symbols:
        files = sorted(NEWS_DIR.glob(f"{sym}_*.parquet"))
        if not files:
            continue
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        if "score" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["datetime"]).dt.tz_convert(None).dt.normalize()
        agg = df.groupby("date").agg(
            sent_mean=("score", "mean"),
            sent_max_neg=("p_negative", "max"),
            n_articles=("score", "size"),
        )
        agg["symbol"] = sym
        rows.append(agg.reset_index())
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "sent_mean", "sent_max_neg", "n_articles"])
    return pd.concat(rows, ignore_index=True)


# --------------------------- 1. market-wide ----------------------------

def market_sentiment_from_rss(scored_rss_path) -> pd.DataFrame:
    """Compute daily market-wide sentiment from the RSS market news file.

    Expected input: a parquet at `data/news/_rss_market.parquet` with the
    raw RSS items from `rss_news.fetch_all_whitelisted()`. We re-score
    titles+summaries with FinBERT here on demand if 'score' missing.
    Returns columns: date, market_sent_mean, market_sent_max_neg, market_n.
    """
    if not scored_rss_path.exists():
        return pd.DataFrame(columns=["date", "market_sent_mean",
                                     "market_sent_max_neg", "market_n"])
    df = pd.read_parquet(scored_rss_path)
    if df.empty:
        return pd.DataFrame(columns=["date", "market_sent_mean",
                                     "market_sent_max_neg", "market_n"])

    if "score" not in df.columns:
        from src.features.sentiment import FinBERTScorer, score_news_dataframe
        df = df.rename(columns={"title": "headline"})  # match expected col
        df = score_news_dataframe(df, scorer=FinBERTScorer(),
                                  text_columns=("headline", "summary"))

    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_convert(None).dt.normalize()
    out = df.groupby("date").agg(
        market_sent_mean=("score", "mean"),
        market_sent_max_neg=("p_negative", "max"),
        market_n=("score", "size"),
    ).reset_index()
    return out


# --------------------------- 2. sector ---------------------------------

def sector_sentiment(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (date, sector) mean + dispersion of stock sentiment.

    Returns columns: date, gics_sector, sector_sent_mean, sector_sent_disp.
    Will be joined back to per-stock rows via the stock's sector.
    """
    if panel.empty:
        return pd.DataFrame(columns=["date", "gics_sector",
                                     "sector_sent_mean", "sector_sent_disp"])

    sp500 = universe.load_sp500()[["symbol", "gics_sector"]]
    merged = panel.merge(sp500, on="symbol", how="left").dropna(subset=["gics_sector"])
    out = merged.groupby(["date", "gics_sector"]).agg(
        sector_sent_mean=("sent_mean", "mean"),
        sector_sent_disp=("sent_mean", "std"),
    ).reset_index()
    out["sector_sent_disp"] = out["sector_sent_disp"].fillna(0.0)
    return out


# --------------------------- 3. peer-correlation-weighted --------------

def peer_sentiment(
    panel: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    top_k_peers: int = 10,
) -> pd.DataFrame:
    """Correlation-weighted mean sentiment of top-k peers (excluding self).

    For each (date, symbol):
        peer_sent_mean = Σ_{p in top-k peers}  corr(self, p) * sent_p,d
                       / Σ corr(self, p)
        peer_sent_max_neg = max(sent_max_neg over those peers)

    Returns columns: date, symbol, peer_sent_mean, peer_sent_max_neg, peer_coverage.
    """
    if panel.empty or correlation_matrix.empty:
        return pd.DataFrame(columns=["date", "symbol",
                                     "peer_sent_mean", "peer_sent_max_neg", "peer_coverage"])

    # Pre-compute peer lists once (top-k by abs correlation, excluding self)
    peers: dict[str, list[tuple[str, float]]] = {}
    for sym in correlation_matrix.columns:
        if sym not in correlation_matrix.index:
            continue
        col = correlation_matrix[sym].drop(labels=[sym], errors="ignore").dropna()
        top = col.reindex(col.abs().sort_values(ascending=False).index).head(top_k_peers)
        peers[sym] = list(zip(top.index, top.values))

    # Index sentiment by (date, symbol) for fast lookup
    pivot_mean = panel.pivot_table(index="date", columns="symbol", values="sent_mean")
    pivot_neg = panel.pivot_table(index="date", columns="symbol", values="sent_max_neg")

    rows = []
    for sym, peer_list in peers.items():
        # only peers we actually have sentiment for
        present = [(p, w) for p, w in peer_list if p in pivot_mean.columns]
        if not present:
            continue
        ps, ws = zip(*present)
        ws_arr = np.array(ws)
        ws_norm = ws_arr / (np.abs(ws_arr).sum() + 1e-9)

        peer_means = pivot_mean[list(ps)].fillna(0.0)
        peer_negs = pivot_neg[list(ps)].fillna(0.0)
        coverage = pivot_mean[list(ps)].notna().sum(axis=1) / len(ps)

        weighted = (peer_means * ws_norm).sum(axis=1)
        max_neg = peer_negs.max(axis=1)

        df = pd.DataFrame({
            "date": peer_means.index,
            "symbol": sym,
            "peer_sent_mean": weighted.values,
            "peer_sent_max_neg": max_neg.values,
            "peer_coverage": coverage.values,
        })
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "peer_sent_mean",
                                     "peer_sent_max_neg", "peer_coverage"])
    return pd.concat(rows, ignore_index=True)


# --------------------------- 4. lags + momentum ------------------------

def sentiment_lags(panel: pd.DataFrame, lags: tuple[int, ...] = (1, 3, 7)) -> pd.DataFrame:
    """Per-symbol sentiment lagged by N business days, plus momentum.

    Returns the input panel with extra columns:
        sent_mean_t1, sent_mean_t3, sent_mean_t7, sent_momentum_7
    """
    if panel.empty:
        return panel
    df = panel.sort_values(["symbol", "date"]).copy()
    for lag in lags:
        df[f"sent_mean_t{lag}"] = df.groupby("symbol")["sent_mean"].shift(lag)
    if 7 in lags:
        df["sent_momentum_7"] = df["sent_mean"] - df["sent_mean_t7"]
    return df.fillna({col: 0.0 for col in df.columns if col.startswith("sent_mean_t")
                      or col == "sent_momentum_7"})


# --------------------------- one-stop builder --------------------------

def build_cross_sentiment_features(
    symbols: list[str],
    correlation_matrix: pd.DataFrame | None = None,
    rss_market_path=None,
) -> pd.DataFrame:
    """Build the full long-format cross-sentiment feature panel.

    Returns columns:
        date, symbol,
        sent_mean, sent_max_neg, n_articles,                 # base (per-symbol)
        sent_mean_t1, sent_mean_t3, sent_mean_t7,            # lags
        sent_momentum_7,                                     # trend
        sector_sent_mean, sector_sent_disp,                  # sector
        peer_sent_mean, peer_sent_max_neg, peer_coverage,    # peers
        market_sent_mean, market_sent_max_neg, market_n      # market

    `correlation_matrix` should come from
    `correlations.correlation_matrix(returns, window=252)`. If None,
    peer features are skipped.
    """
    panel = _per_symbol_sentiment_panel(symbols)
    if panel.empty:
        return panel

    # 4. lags
    panel = sentiment_lags(panel)

    # 2. sector
    sect = sector_sentiment(panel)
    sp500 = universe.load_sp500()[["symbol", "gics_sector"]]
    panel = panel.merge(sp500, on="symbol", how="left")
    panel = panel.merge(sect, on=["date", "gics_sector"], how="left")

    # 3. peers
    if correlation_matrix is not None:
        peer = peer_sentiment(panel, correlation_matrix)
        panel = panel.merge(peer, on=["date", "symbol"], how="left")

    # 1. market
    if rss_market_path is not None:
        from pathlib import Path
        mkt = market_sentiment_from_rss(Path(rss_market_path))
        panel = panel.merge(mkt, on="date", how="left")

    fill_cols = [c for c in panel.columns
                 if c.startswith(("sent_", "sector_", "peer_", "market_"))]
    panel[fill_cols] = panel[fill_cols].fillna(0.0)
    return panel.drop(columns=["gics_sector"], errors="ignore")
