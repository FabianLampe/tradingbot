"""Company news ingestion — Finnhub primary, Alpha Vantage fallback.

**Finnhub** `company-news` is the workhorse: free tier gives 60 calls/min
and ~1 year of history, one call per ticker per date range. When no
Finnhub key is configured (or a symbol comes back empty) we fall back to
**Alpha Vantage** `NEWS_SENTIMENT` — only 25 calls/day on the free tier,
so it is a stopgap for a handful of tickers, not a bulk backfill.

Both providers are normalised to one schema, so downstream code never has
to know where a row came from:

    news_id   str                    stable id -> idempotent dedupe
    symbol    str
    datetime  datetime64[ns, UTC]    tz-aware, always UTC
    headline  str
    summary   str
    source    str                    outlet as reported by the provider
    url       str
    category  str
    provider  str                    "finnhub" | "alphavantage"

`datetime` is deliberately **tz-aware**: `features.sentiment.aggregate_daily`
calls `.dt.tz_convert(None)` on it and `runtime.premarket` compares it
against a tz-aware market-close timestamp — a naive column breaks both.

Outlets listed under `news_outlets_blacklist` in `config/whitelist.yaml`
(Benzinga, Zacks, …) are dropped here rather than downstream: they publish
near-identical templated articles for dozens of tickers, which would
otherwise dominate the per-day article counts FinBERT aggregates over.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Iterable, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm

import config
from src.data._common import (
    RateLimiter,
    clean_text,
    empty_frame,
    finalize,
    outlet_key,
    stable_id,
    to_utc,
    with_retries,
)

log = logging.getLogger("trading_bot.data.news")

NEWS_COLUMNS: tuple[str, ...] = (
    "news_id", "symbol", "datetime", "headline", "summary",
    "source", "url", "category", "provider",
)

# Finnhub free tier allows 60/min; stay just under to survive clock skew.
FINNHUB_CALLS_PER_MINUTE = 55
FINNHUB_MAX_DAYS_PER_CALL = 90  # longer ranges get silently truncated

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"


def empty_news_frame() -> pd.DataFrame:
    """Correctly-typed empty frame with the canonical news schema."""
    return empty_frame(NEWS_COLUMNS)


# ------------------------------ whitelist -------------------------------

@lru_cache(maxsize=1)
def blacklisted_outlets() -> frozenset[str]:
    """Outlet keys to drop, from `whitelist.yaml: news_outlets_blacklist`."""
    entries = config.load_whitelist().get("news_outlets_blacklist") or []
    return frozenset(outlet_key(e) for e in entries if e)


def drop_blacklisted_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Remove articles whose outlet (by name *or* URL host) is blacklisted."""
    blocked = blacklisted_outlets()
    if df.empty or not blocked:
        return df
    by_source = df["source"].map(outlet_key).isin(blocked)
    by_url = df["url"].map(outlet_key).isin(blocked)
    keep = ~(by_source | by_url)
    dropped = int((~keep).sum())
    if dropped:
        log.debug("Dropped %d blacklisted articles", dropped)
    return df[keep].reset_index(drop=True)


# ------------------------------- Finnhub --------------------------------

@lru_cache(maxsize=1)
def _finnhub_client():
    """Cached Finnhub client, or None when no key is configured."""
    if not config.FINNHUB_API_KEY:
        return None
    import finnhub  # imported lazily so the module works without the SDK

    return finnhub.Client(api_key=config.FINNHUB_API_KEY)


def _resolve_window(
    days_back: int,
    start: str | date | None,
    end: str | date | None,
) -> tuple[date, date]:
    """Normalise (days_back | start/end) into a concrete [start, end] pair."""
    end_d = pd.Timestamp(end).date() if end else datetime.now(tz=timezone.utc).date()
    if start is not None:
        start_d = pd.Timestamp(start).date()
    else:
        start_d = end_d - timedelta(days=max(0, days_back))
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    return start_d, end_d


def _chunk_window(start: date, end: date, max_days: int) -> list[tuple[date, date]]:
    """Split a date range into provider-sized chunks."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=max_days - 1), end)
        chunks.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return chunks


def _finnhub_rows(symbol: str, start: date, end: date, limiter: RateLimiter | None) -> list[dict]:
    client = _finnhub_client()
    if client is None:
        return []
    rows: list[dict] = []
    for chunk_start, chunk_end in _chunk_window(start, end, FINNHUB_MAX_DAYS_PER_CALL):
        if limiter is not None:
            limiter.wait()

        def _call() -> list[dict]:
            return client.company_news(
                symbol,
                _from=chunk_start.isoformat(),
                to=chunk_end.isoformat(),
            ) or []

        articles = with_retries(
            _call,
            on_error=lambda attempt, e: log.warning(
                "[%s] Finnhub attempt %d failed: %s", symbol, attempt, e
            ),
        )
        for art in articles:
            ts = to_utc(art.get("datetime"))
            if ts is None:
                continue
            art_id = art.get("id")
            rows.append({
                "news_id": (f"finnhub:{art_id}" if art_id
                            else stable_id("finnhub", symbol, art.get("url"), art.get("headline"))),
                "symbol": symbol,
                "datetime": ts,
                "headline": clean_text(art.get("headline"), max_chars=512),
                "summary": clean_text(art.get("summary")),
                "source": clean_text(art.get("source"), max_chars=128),
                "url": str(art.get("url") or ""),
                "category": clean_text(art.get("category"), max_chars=64),
                "provider": "finnhub",
            })
    return rows


# ---------------------------- Alpha Vantage -----------------------------

def _alphavantage_rows(symbol: str, start: date, end: date, limit: int = 1000) -> list[dict]:
    """Fallback provider. Returns [] on any error — never raises upward."""
    if not config.ALPHAVANTAGE_API_KEY:
        return []
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "time_from": f"{start:%Y%m%d}T0000",
        "time_to": f"{end:%Y%m%d}T2359",
        "limit": str(limit),
        "sort": "LATEST",
        "apikey": config.ALPHAVANTAGE_API_KEY,
    }
    try:
        payload = with_retries(
            lambda: requests.get(ALPHAVANTAGE_URL, params=params, timeout=30).json(),
            attempts=2,
            on_error=lambda attempt, e: log.warning(
                "[%s] Alpha Vantage attempt %d failed: %s", symbol, attempt, e
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] Alpha Vantage unavailable: %s", symbol, e)
        return []

    # Rate-limit / error responses come back as 200 with a Note/Information key.
    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            log.warning("[%s] Alpha Vantage: %s", symbol, payload[key])
            return []

    rows: list[dict] = []
    for art in payload.get("feed") or []:
        ts = to_utc(art.get("time_published"))
        if ts is None:
            continue
        topics = art.get("topics") or []
        rows.append({
            "news_id": stable_id("av", art.get("url"), art.get("title")),
            "symbol": symbol,
            "datetime": ts,
            "headline": clean_text(art.get("title"), max_chars=512),
            "summary": clean_text(art.get("summary")),
            "source": clean_text(art.get("source"), max_chars=128),
            "url": str(art.get("url") or ""),
            "category": clean_text(
                ", ".join(str(t.get("topic", "")) for t in topics if isinstance(t, dict)),
                max_chars=64,
            ),
            "provider": "alphavantage",
        })
    return rows


# ------------------------------ public API ------------------------------

def fetch_company_news(
    symbol: str,
    days_back: int = 7,
    start: str | date | None = None,
    end: str | date | None = None,
    drop_blacklisted: bool = True,
    allow_fallback: bool = True,
    limiter: RateLimiter | None = None,
) -> pd.DataFrame:
    """Fetch news for one ticker.

    Window is `[today - days_back, today]` unless `start`/`end` are given.
    Falls back to Alpha Vantage when Finnhub yields nothing and a key is
    configured. Returns an empty (but correctly typed) frame if no provider
    is available — callers should not have to special-case missing keys.
    """
    start_d, end_d = _resolve_window(days_back, start, end)

    rows = _finnhub_rows(symbol, start_d, end_d, limiter)
    if not rows and allow_fallback:
        rows = _alphavantage_rows(symbol, start_d, end_d)
    if not rows and not config.FINNHUB_API_KEY and not config.ALPHAVANTAGE_API_KEY:
        log.warning("No news provider configured — set FINNHUB_API_KEY in .env")

    df = finalize(rows, NEWS_COLUMNS)
    # Providers ignore the range boundaries now and then; enforce them.
    if not df.empty:
        lo = pd.Timestamp(start_d, tz="UTC")
        hi = pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(days=1)
        df = df[(df["datetime"] >= lo) & (df["datetime"] < hi)].reset_index(drop=True)
    return drop_blacklisted_sources(df) if drop_blacklisted else df


def fetch_bulk_company_news(
    symbols: Sequence[str],
    days_back: int = 7,
    start: str | date | None = None,
    end: str | date | None = None,
    calls_per_minute: int = FINNHUB_CALLS_PER_MINUTE,
    drop_blacklisted: bool = True,
    allow_fallback: bool = False,
    show_progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch news for many tickers, sharing one rate limiter.

    A failing symbol yields an empty frame rather than killing the batch —
    the daily pipeline must survive one bad ticker out of 500.

    `allow_fallback` defaults to False here: Alpha Vantage's 25 calls/day
    would be burned by the first two dozen tickers of a bulk run.
    """
    limiter = RateLimiter(calls_per_minute)
    out: dict[str, pd.DataFrame] = {}
    iterator: Iterable[str] = symbols
    if show_progress:
        iterator = tqdm(symbols, desc="News", unit="ticker")

    for sym in iterator:
        try:
            out[sym] = fetch_company_news(
                sym,
                days_back=days_back,
                start=start,
                end=end,
                drop_blacklisted=drop_blacklisted,
                allow_fallback=allow_fallback,
                limiter=limiter,
            )
        except Exception as e:  # noqa: BLE001
            log.error("[%s] news fetch failed: %s", sym, e)
            out[sym] = empty_news_frame()
    return out


def merge_news(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """Idempotent merge of a fresh pull into a cached year-file.

    Newer rows win on `news_id` collisions — providers do edit headlines
    and back-fill summaries after publication.
    """
    if existing is None or existing.empty:
        return incoming.reset_index(drop=True)
    if incoming is None or incoming.empty:
        return existing.reset_index(drop=True)
    combined = pd.concat([existing, incoming], ignore_index=True)
    if "news_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["news_id"], keep="last")
    if "datetime" in combined.columns:
        combined = combined.sort_values("datetime")
    return combined.reset_index(drop=True)
