"""Intraday OHLCV bars via yfinance.

The daily pipeline holds positions for days; news moves prices over minutes to
hours. This module supplies the bars needed to test the shorter horizon where
a news signal could plausibly still be worth something.

**Yahoo's history limits are the binding constraint** and the reason `1h` is
the default here rather than `1m`:

    1m                 ~7 days
    2m / 5m / 15m /
    30m / 90m          ~60 days
    1h / 60m           ~730 days
    1d and coarser     decades

A minute-level strategy cannot be backtested on this source at all — seven
days is not a sample, it is an anecdote. Hourly bars over two years give
roughly 3,500 observations per symbol, which is enough to reject a bad idea
cheaply before paying for real intraday history (Polygon, Databento, IQFeed).

Storage is one Parquet per (symbol, interval) under `data/prices_intraday/`,
separate from the daily cache so the two can never be confused.

The index is **tz-aware UTC**, unlike the daily cache. A 09:30 bar only means
something relative to the exchange session, and a naive index would silently
misalign bars across daylight-saving changes.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from tqdm.auto import tqdm

from config import INTRADAY_DIR
from src.data._common import with_retries
from src.data.prices import PRICE_COLUMNS, _normalise, empty_price_frame, merge_prices

log = logging.getLogger("trading_bot.data.intraday")

#: Maximum lookback Yahoo serves per interval, in days.
INTERVAL_MAX_DAYS: dict[str, int] = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "90m": 60,
    "60m": 730, "1h": 730,
}

#: Bars per regular US trading session (6.5 h), used to size warmup windows.
BARS_PER_SESSION: dict[str, float] = {
    "1m": 390, "2m": 195, "5m": 78, "15m": 26, "30m": 13,
    "60m": 6.5, "1h": 6.5, "90m": 4.33,
}

DEFAULT_INTERVAL = "1h"


def supported_intervals() -> list[str]:
    return sorted(INTERVAL_MAX_DAYS)


def _check_interval(interval: str) -> None:
    if interval not in INTERVAL_MAX_DAYS:
        raise ValueError(
            f"Unsupported interval {interval!r}. Yahoo serves {supported_intervals()}."
        )


def intraday_path(symbol: str, interval: str = DEFAULT_INTERVAL) -> Path:
    return INTRADAY_DIR / f"{symbol.upper()}_{interval}.parquet"


def read_intraday(symbol: str, interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    path = intraday_path(symbol, interval)
    if not path.exists():
        raise FileNotFoundError(f"No {interval} bars cached for {symbol} at {path}")
    df = pd.read_parquet(path)
    if not df.empty and getattr(df.index, "tz", None) is None:
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC")
    df.index.name = "timestamp"
    return df


def write_intraday(symbol: str, interval: str, df: pd.DataFrame) -> Path:
    path = intraday_path(symbol, interval)
    df.to_parquet(path, index=True)
    return path


def download_intraday(
    symbol: str,
    interval: str = DEFAULT_INTERVAL,
    days_back: int | None = None,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Download intraday bars for one symbol.

    `days_back` is clamped to what Yahoo actually serves for the interval —
    asking for more silently returns a short frame, which is worse than being
    told, so we clamp loudly instead.
    """
    import yfinance as yf

    _check_interval(interval)
    max_days = INTERVAL_MAX_DAYS[interval]
    if days_back is None:
        days_back = max_days
    if days_back > max_days:
        log.warning(
            "%s bars go back at most %d days on Yahoo; clamping your %d-day request.",
            interval, max_days, days_back,
        )
        days_back = max_days

    if start is None:
        start = (datetime.now(tz=timezone.utc).date() - timedelta(days=days_back)).isoformat()

    def _call():
        return yf.download(
            symbol,
            start=str(start),
            end=str(end) if end else None,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    try:
        raw = with_retries(
            _call,
            attempts=3,
            on_error=lambda attempt, e: log.warning(
                "[%s %s] attempt %d failed: %s", symbol, interval, attempt, e
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.error("[%s %s] download failed: %s", symbol, interval, e)
        return empty_price_frame(intraday=True)

    return _normalise(raw, symbol, intraday=True)


def download_and_cache_intraday(
    symbols: Iterable[str],
    interval: str = DEFAULT_INTERVAL,
    days_back: int | None = None,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download and merge into `data/prices_intraday/{SYMBOL}_{interval}.parquet`.

    Merges like the daily cache: because Yahoo's window rolls forward, running
    this regularly is the *only* way to accumulate more history than the
    lookback limit allows. Miss a few weeks and that stretch is gone for good.
    """
    _check_interval(interval)
    symbols = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    iterator: Iterable[str] = symbols
    if show_progress:
        iterator = tqdm(symbols, desc=f"Intraday {interval}", unit="ticker")

    cached: dict[str, pd.DataFrame] = {}
    for sym in iterator:
        fresh = download_intraday(sym, interval=interval, days_back=days_back)
        if fresh.empty:
            log.warning("[%s] no %s bars returned", sym, interval)
            continue
        existing = read_intraday(sym, interval) if intraday_path(sym, interval).exists() else None
        merged = merge_prices(existing, fresh)
        merged.index.name = "timestamp"
        write_intraday(sym, interval, merged)
        cached[sym] = merged

    log.info("Cached %s bars for %d/%d symbols", interval, len(cached), len(symbols))
    return cached


def build_bar_panel(
    symbols: Sequence[str],
    interval: str = DEFAULT_INTERVAL,
    column: str = "close",
) -> pd.DataFrame:
    """Wide frame: index=timestamp, columns=symbol. Missing bars stay NaN."""
    series = {}
    for sym in symbols:
        try:
            df = read_intraday(sym, interval)
        except FileNotFoundError:
            continue
        if not df.empty and column in df.columns:
            series[sym] = df[column]
    if not series:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    panel = pd.DataFrame(series).sort_index()
    panel.index.name = "timestamp"
    return panel


def coverage(symbols: Sequence[str], interval: str = DEFAULT_INTERVAL) -> pd.DataFrame:
    """What is actually cached per symbol — bars, span, gaps.

    Worth checking before any intraday backtest: Yahoo's intraday history is
    patchy, and a symbol with 300 bars will quietly produce a meaningless
    result rather than an error.
    """
    rows = []
    for sym in symbols:
        try:
            df = read_intraday(sym, interval)
        except FileNotFoundError:
            rows.append({"symbol": sym, "bars": 0, "first": None, "last": None,
                         "sessions": 0})
            continue
        rows.append({
            "symbol": sym,
            "bars": len(df),
            "first": df.index.min() if len(df) else None,
            "last": df.index.max() if len(df) else None,
            "sessions": int(df.index.normalize().nunique()) if len(df) else 0,
        })
    return pd.DataFrame(rows)
