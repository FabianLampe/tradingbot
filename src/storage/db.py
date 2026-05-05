"""Parquet-backed storage helpers.

We use Parquet (one file per ticker) rather than SQLite/Postgres because:
  - Columnar => fast aggregations across tickers.
  - Good compression (5–10x vs CSV).
  - Native pandas + pyarrow support, no daemon to run.
  - Trivial to ship to the GPU server (just rsync the directory).

For news we use one file per ticker per year to keep partitions small.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import NEWS_DIR, PRICES_DIR


# ---------- prices ----------

def price_path(symbol: str) -> Path:
    return PRICES_DIR / f"{symbol}.parquet"


def write_prices(symbol: str, df: pd.DataFrame) -> Path:
    """Persist OHLCV history for a symbol. Overwrites existing file."""
    path = price_path(symbol)
    df.to_parquet(path, index=True)
    return path


def read_prices(symbol: str) -> pd.DataFrame:
    path = price_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No price data for {symbol} at {path}")
    return pd.read_parquet(path)


def read_all_prices(symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every cached symbol (or a subset) into a dict."""
    if symbols is None:
        symbols = [p.stem for p in PRICES_DIR.glob("*.parquet")]
    return {s: read_prices(s) for s in symbols if price_path(s).exists()}


# ---------- news ----------

def news_path(symbol: str, year: int) -> Path:
    return NEWS_DIR / f"{symbol}_{year}.parquet"


def write_news(symbol: str, year: int, df: pd.DataFrame) -> Path:
    path = news_path(symbol, year)
    df.to_parquet(path, index=False)
    return path


def read_news(symbol: str, year: int | None = None) -> pd.DataFrame:
    """Read news for a symbol. If year is None, concatenate all cached years."""
    if year is not None:
        return pd.read_parquet(news_path(symbol, year))
    files = sorted(NEWS_DIR.glob(f"{symbol}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No news cached for {symbol}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
