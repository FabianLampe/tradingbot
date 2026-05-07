"""Build the supervised learning dataset.

Joins three feature families on (symbol, date) and adds a forward-return target.
Output is one tidy DataFrame ready for sklearn/XGBoost.

  features = technical (per-stock) ⊕ sentiment (per-stock-day) ⊕ macro (per-day)
  target   = forward log-return over `horizon_days` business days
  label    = 1 if forward return > +`up_thresh`, -1 if < -`down_thresh`, else 0

Convention: each row uses **only information available at its date**. The
target is the *future* return — kept in a separate column so we never
accidentally feed it to the model. Anti-leakage discipline is on you,
but the structure makes it harder to mess up.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from config import MACRO_DIR, NEWS_DIR
from src.features import returns as ret_mod
from src.features import technical
from src.storage import db


def _load_cross_sentiment(symbols: list[str]) -> pd.DataFrame | None:
    """Build (or return None) the cross-stock + sector + market + lag sentiment panel.

    Skipped silently if any of (sentiment, returns, sp500-list) are missing.
    """
    try:
        from src.features import correlations, cross_sentiment, returns as ret
        from config import NEWS_DIR
        closes = ret.build_close_panel(symbols)
        log_ret = ret.log_returns(closes)
        corr = correlations.correlation_matrix(log_ret, window=252)
        rss_path = NEWS_DIR / "_rss_market.parquet"
        return cross_sentiment.build_cross_sentiment_features(
            symbols,
            correlation_matrix=corr,
            rss_market_path=rss_path if rss_path.exists() else None,
        )
    except Exception:
        return None


def _load_macro_panel() -> pd.DataFrame | None:
    path = MACRO_DIR / "fred_panel.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def _load_sentiment_for_symbol(symbol: str) -> pd.DataFrame | None:
    """Read all cached sentiment-aggregated news for a symbol.

    Expects files at data/news/{symbol}_{year}.parquet that already contain
    the FinBERT score columns (run notebooks/03 first, or call
    sentiment.score_news_dataframe before save).
    """
    files = sorted(NEWS_DIR.glob(f"{symbol}_*.parquet"))
    if not files:
        return None
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    if "score" not in df.columns:
        return None  # not yet scored
    df["date"] = pd.to_datetime(df["datetime"]).dt.tz_convert(None).dt.normalize()
    return df.groupby("date").agg(
        n_articles=("score", "size"),
        sent_mean=("score", "mean"),
        sent_max_neg=("p_negative", "max"),
        sent_std=("score", "std"),
    ).fillna({"sent_std": 0.0})


def build_per_symbol(
    symbol: str,
    horizon_days: int = 5,
    up_thresh: float = 0.01,
    down_thresh: float = 0.01,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Build supervised rows for one symbol. None if no data."""
    try:
        prices = db.read_prices(symbol)
    except FileNotFoundError:
        return None
    if prices.empty:
        return None

    # technical features
    feat = technical.add_technical_features(prices)
    feat["symbol"] = symbol

    # sentiment (left-join — many days will be NaN; we'll fill with 0)
    sent = _load_sentiment_for_symbol(symbol)
    if sent is not None:
        feat = feat.join(sent, how="left")
        feat[["n_articles", "sent_mean", "sent_max_neg", "sent_std"]] = (
            feat[["n_articles", "sent_mean", "sent_max_neg", "sent_std"]].fillna(0.0)
        )
    else:
        for c in ("n_articles", "sent_mean", "sent_max_neg", "sent_std"):
            feat[c] = 0.0

    # macro (broadcast to every row by date)
    if macro is not None:
        feat = feat.join(macro, how="left").ffill()

    # target: forward log-return
    close = feat["adj_close"]
    feat["fwd_return"] = np.log(close.shift(-horizon_days) / close)
    feat["fwd_class"] = 0
    feat.loc[feat["fwd_return"] > up_thresh, "fwd_class"] = 1
    feat.loc[feat["fwd_return"] < -down_thresh, "fwd_class"] = -1

    # drop warmup NaNs (200-day EMA is the longest lookback)
    feat = feat.dropna(subset=technical.FEATURE_COLUMNS)
    return feat.reset_index().rename(columns={"index": "date"})


def build_dataset(
    symbols: Iterable[str],
    horizon_days: int = 5,
    up_thresh: float = 0.01,
    down_thresh: float = 0.01,
    include_cross_sentiment: bool = True,
) -> pd.DataFrame:
    """Build the full long-format supervised dataset for many symbols.

    If `include_cross_sentiment` is True (default), also joins peer-,
    sector-, market-, and lag-sentiment features built by the
    `cross_sentiment` module. Skipped silently if dependencies are missing.
    """
    symbols = list(symbols)
    macro = _load_macro_panel()
    rows: list[pd.DataFrame] = []
    for sym in tqdm(symbols, desc="Building dataset"):
        df = build_per_symbol(
            sym,
            horizon_days=horizon_days,
            up_thresh=up_thresh,
            down_thresh=down_thresh,
            macro=macro,
        )
        if df is not None and not df.empty:
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)

    if include_cross_sentiment:
        cross = _load_cross_sentiment(symbols)
        if cross is not None and not cross.empty:
            cross["date"] = pd.to_datetime(cross["date"])
            full["date"] = pd.to_datetime(full["date"])
            # avoid duplicate columns from base sentiment
            keep_cols = ["date", "symbol"] + [
                c for c in cross.columns
                if c not in full.columns and c not in ("date", "symbol")
            ]
            full = full.merge(cross[keep_cols], on=["date", "symbol"], how="left")
            new_cols = [c for c in keep_cols if c not in ("date", "symbol")]
            full[new_cols] = full[new_cols].fillna(0.0)

    return full


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The columns the model should see. Excludes id/target/raw-OHLCV."""
    exclude = {
        "date", "symbol", "open", "high", "low", "close", "adj_close", "volume",
        "fwd_return", "fwd_class",
    }
    return [c for c in df.columns if c not in exclude]
