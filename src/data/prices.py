"""OHLCV price ingestion via yfinance.

Output schema — one Parquet file per ticker under `data/prices/`:

    index    date         DatetimeIndex, tz-naive, normalised to midnight
    open, high, low, close, adj_close   float
    volume                              int

`close` vs `adj_close` is not interchangeable and both are kept on purpose:
`features.technical` computes indicators on raw `close` (yfinance already
split-adjusts OHLC, and volume must line up with unadjusted prices), while
`features.returns` and the forward-return target use `adj_close` so
dividends do not show up as phantom gaps.

The index is tz-naive because everything downstream compares it against
naive timestamps — `runtime.daily` books outcomes with
`pd.Timestamp(asof_date)`, and `features.build_dataset` joins the macro
panel on plain dates. A tz-aware index raises on both.

**Caching merges, it does not overwrite.** `runtime.daily` re-downloads only
the last 30 days on every run; overwriting would truncate a 15-year history
to one month on the first scheduled run.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

import pandas as pd
from tqdm.auto import tqdm

from config import DEFAULT_HISTORY_YEARS
from src.data._common import with_retries
from src.storage import db

log = logging.getLogger("trading_bot.data.prices")

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "adj_close", "volume")

_COLUMN_ALIASES = {
    "open": "open", "high": "high", "low": "low", "close": "close",
    "adj close": "adj_close", "adjclose": "adj_close", "adj_close": "adj_close",
    "volume": "volume",
}


def empty_price_frame() -> pd.DataFrame:
    """Correctly-typed empty frame with the canonical price schema."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in PRICE_COLUMNS})
    df["volume"] = pd.Series(dtype="int64")
    df.index = pd.DatetimeIndex([], name="date")
    return df


def _default_start(years: int = DEFAULT_HISTORY_YEARS) -> str:
    return (datetime.now(tz=timezone.utc).date() - timedelta(days=365 * years)).isoformat()


def _normalise(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """yfinance frame -> canonical schema. Empty frame if unusable."""
    if raw is None or len(raw) == 0:
        return empty_price_frame()

    df = raw.copy()

    # yfinance returns MultiIndex columns for multi-ticker downloads, and in
    # recent versions for single tickers too. Pick this symbol's slice.
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if symbol in df.columns.get_level_values(level):
                df = df.xs(symbol, axis=1, level=level)
                break
        else:
            df.columns = df.columns.get_level_values(0)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [_COLUMN_ALIASES.get(str(c).strip().lower(), str(c).strip().lower())
                  for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]

    # auto_adjust=True drops "Adj Close"; then close is already adjusted.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    missing = [c for c in PRICE_COLUMNS if c not in df.columns]
    if missing:
        log.warning("[%s] missing columns %s — skipping", symbol, missing)
        return empty_price_frame()

    df = df[list(PRICE_COLUMNS)]

    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    # Pin the resolution: pandas infers us/ns depending on the input, and a
    # mixed-unit index breaks the join against the macro panel downstream.
    df.index = pd.DatetimeIndex(idx).normalize().as_unit("ns")
    df.index.name = "date"

    df = df[df.index.notna()]
    df = df.dropna(subset=["close"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if df.empty:
        return empty_price_frame()

    for col in ("open", "high", "low", "close", "adj_close"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df


def merge_prices(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Union two price frames on the date index; newer rows win.

    This is what keeps the daily 30-day refresh from truncating history.
    """
    if existing is None or existing.empty:
        return incoming
    if incoming is None or incoming.empty:
        return existing
    combined = pd.concat([existing, incoming])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.index.name = "date"
    return combined


def download_one(
    symbol: str,
    start: str | date | None = None,
    end: str | date | None = None,
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Download OHLCV history for one ticker.

    Defaults to `DEFAULT_HISTORY_YEARS` of history — long enough to span the
    2008 crisis, the 2020 crash and the 2022 rate-hike cycle.
    """
    import yfinance as yf

    start = start or _default_start(years)

    def _call():
        return yf.download(
            symbol,
            start=str(start),
            end=str(end) if end else None,
            auto_adjust=False,   # keep raw close AND adj_close
            progress=False,
            threads=False,
        )

    try:
        raw = with_retries(
            _call,
            attempts=3,
            on_error=lambda attempt, e: log.warning(
                "[%s] download attempt %d failed: %s", symbol, attempt, e
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.error("[%s] download failed: %s", symbol, e)
        return empty_price_frame()
    return _normalise(raw, symbol)


def download_batch(
    symbols: Sequence[str],
    start: str | date | None = None,
    end: str | date | None = None,
    years: int = DEFAULT_HISTORY_YEARS,
) -> dict[str, pd.DataFrame]:
    """Download several tickers in one yfinance call (much faster than a loop).

    Falls back to per-symbol downloads if the batch call fails outright, so
    one delisted ticker cannot take down the whole batch.
    """
    import yfinance as yf

    symbols = list(symbols)
    if not symbols:
        return {}
    start = start or _default_start(years)

    try:
        raw = yf.download(
            symbols,
            start=str(start),
            end=str(end) if end else None,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Batch download failed (%s) — falling back to per-symbol", e)
        return {s: download_one(s, start=start, end=end) for s in symbols}

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _normalise(raw, sym)
        if df.empty and len(symbols) > 1:
            df = download_one(sym, start=start, end=end)   # retry this one alone
        out[sym] = df
    return out


def benchmark_cache_name(symbol: str) -> str:
    """Filename-safe stem for an index ticker ('^GSPC' -> '_GSPC').

    Index tickers start with '^', which is awkward in a path and would also
    make the index look like a tradeable symbol in `read_all_prices`.
    """
    return "_" + str(symbol).lstrip("^").upper()


def download_benchmark(
    symbol: str | None = None,
    start: str | date | None = None,
    years: int = DEFAULT_HISTORY_YEARS,
) -> pd.DataFrame:
    """Download and cache the benchmark index (default: `config.DEFAULT_BENCHMARK`).

    Stored under `benchmark_cache_name(symbol)` so `src.model.backtest` can
    find it for the benchmark-relative metrics.
    """
    from config import DEFAULT_BENCHMARK

    symbol = symbol or DEFAULT_BENCHMARK
    df = download_one(symbol, start=start, years=years)
    if df.empty:
        log.warning("[%s] benchmark download returned nothing", symbol)
        return df
    name = benchmark_cache_name(symbol)
    existing = db.read_prices(name) if db.price_path(name).exists() else None
    merged = merge_prices(existing, df)
    db.write_prices(name, merged)
    log.info("Benchmark %s cached as %s (%d rows)", symbol, name, len(merged))
    return merged


def download_and_cache(
    symbols: Iterable[str],
    start: str | date | None = None,
    end: str | date | None = None,
    years: int = DEFAULT_HISTORY_YEARS,
    batch_size: int = 50,
    show_progress: bool = True,
    backfill_if_missing: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download and persist to `data/prices/{SYMBOL}.parquet`.

    Returns `{symbol: frame}` for the tickers that produced data — callers
    diff that against their input to find failures
    (`set(symbols) - set(cached)`).

    Existing cached history is **merged**, never replaced, so an incremental
    refresh extends the file instead of truncating it.

    `backfill_if_missing` (default on) fetches full history for a symbol that
    has no cache yet, ignoring `start` for that one symbol. `runtime.daily`
    refreshes only the last 30 days; without this, the first scheduled run on
    a fresh install would seed 30-day files, and the 200-day EMA warmup in
    `features.technical` would then drop every row — an empty dataset with no
    obvious cause. Set it False to cache exactly the window you asked for.
    """
    symbols = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
    if not symbols:
        return {}

    uncached = {s for s in symbols if not db.price_path(s).exists()}
    if backfill_if_missing and uncached and start is not None:
        log.info("Backfilling full history for %d uncached symbol(s)", len(uncached))
    else:
        uncached = set()

    batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
    iterator: Iterable[list[str]] = batches
    if show_progress:
        iterator = tqdm(batches, desc="Prices", unit="batch")

    cached: dict[str, pd.DataFrame] = {}
    for batch in iterator:
        incremental = [s for s in batch if s not in uncached]
        downloaded: dict[str, pd.DataFrame] = {}
        if incremental:
            downloaded.update(download_batch(incremental, start=start, end=end, years=years))
        if batch_backfill := [s for s in batch if s in uncached]:
            downloaded.update(download_batch(batch_backfill, start=None, end=end, years=years))

        for sym, fresh in downloaded.items():
            if fresh.empty:
                log.warning("[%s] no price data returned", sym)
                continue
            try:
                existing = db.read_prices(sym)
            except FileNotFoundError:
                existing = None
            merged = merge_prices(existing, fresh)
            db.write_prices(sym, merged)
            cached[sym] = merged

    log.info("Cached prices for %d/%d symbols", len(cached), len(symbols))
    return cached
