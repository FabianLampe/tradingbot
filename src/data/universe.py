"""Trading universe — S&P 500 constituents plus a curated ETF list.

Two universes, deliberately separate:

  - **ETFs** (`etf_symbols`) — ~30 hand-picked funds spanning the broad
    market, all 11 GICS sectors, duration, credit, commodities and
    international. Small enough that a full pipeline run takes minutes,
    which makes it the default for testing and backtests.
  - **S&P 500** (`symbols`) — scraped from Wikipedia and cached. ~10x
    slower end to end, for production runs.

The S&P 500 list is cached at `data/meta/sp500.parquet` and only re-fetched
with `refresh=True`. That keeps a Wikipedia outage or layout change from
breaking every run, and makes the universe reproducible between runs.

Ticker convention: Wikipedia writes class shares with a dot (BRK.B), Yahoo
Finance wants a hyphen (BRK-B). We normalise to the Yahoo form here, since
that is what `data.prices` feeds to yfinance.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

from config import META_DIR

log = logging.getLogger("trading_bot.data.universe")

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CACHE = META_DIR / "sp500.parquet"

SP500_COLUMNS: tuple[str, ...] = (
    "symbol", "security", "gics_sector", "gics_sub_industry", "cik",
)

# Curated ETF universe. One fund per exposure — no near-duplicates (SPY and
# VOO track the same index; keeping both would double-count the same signal
# in every correlation and sector aggregate).
ETFS: dict[str, tuple[str, str]] = {
    # symbol: (name, bucket)
    "SPY":  ("SPDR S&P 500", "broad"),
    "QQQ":  ("Invesco Nasdaq 100", "broad"),
    "DIA":  ("SPDR Dow Jones Industrial", "broad"),
    "IWM":  ("iShares Russell 2000", "broad"),
    "VTI":  ("Vanguard Total Stock Market", "broad"),
    "XLK":  ("Technology Select Sector", "sector"),
    "XLF":  ("Financial Select Sector", "sector"),
    "XLE":  ("Energy Select Sector", "sector"),
    "XLV":  ("Health Care Select Sector", "sector"),
    "XLI":  ("Industrial Select Sector", "sector"),
    "XLY":  ("Consumer Discretionary Select Sector", "sector"),
    "XLP":  ("Consumer Staples Select Sector", "sector"),
    "XLU":  ("Utilities Select Sector", "sector"),
    "XLB":  ("Materials Select Sector", "sector"),
    "XLRE": ("Real Estate Select Sector", "sector"),
    "XLC":  ("Communication Services Select Sector", "sector"),
    "EFA":  ("iShares MSCI EAFE", "international"),
    "EEM":  ("iShares MSCI Emerging Markets", "international"),
    "VGK":  ("Vanguard FTSE Europe", "international"),
    "EWJ":  ("iShares MSCI Japan", "international"),
    "TLT":  ("iShares 20+ Year Treasury", "bonds"),
    "IEF":  ("iShares 7-10 Year Treasury", "bonds"),
    "SHY":  ("iShares 1-3 Year Treasury", "bonds"),
    "LQD":  ("iShares Investment Grade Corp", "bonds"),
    "HYG":  ("iShares High Yield Corp", "bonds"),
    "GLD":  ("SPDR Gold Shares", "commodities"),
    "SLV":  ("iShares Silver Trust", "commodities"),
    "USO":  ("United States Oil Fund", "commodities"),
    "DBC":  ("Invesco DB Commodity Index", "commodities"),
    "VNQ":  ("Vanguard Real Estate", "real_estate"),
}


def normalise_symbol(symbol: str) -> str:
    """Wikipedia/exchange ticker -> Yahoo Finance form (BRK.B -> BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def _normalise_wiki_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Wikipedia's column labels onto our schema."""
    renamed = {}
    for col in raw.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = {
            "symbol": "symbol",
            "security": "security",
            "gics_sector": "gics_sector",
            "gics_sub-industry": "gics_sub_industry",
            "gics_sub_industry": "gics_sub_industry",
            "cik": "cik",
        }.get(key, key)
    df = raw.rename(columns=renamed)

    missing = [c for c in ("symbol", "gics_sector") if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Wikipedia table is missing {missing} — the page layout changed. "
            f"Got columns: {list(df.columns)}"
        )
    for col in SP500_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[list(SP500_COLUMNS)].copy()
    df["symbol"] = df["symbol"].map(normalise_symbol)
    df["gics_sector"] = df["gics_sector"].astype(str).str.strip()
    df = df[df["symbol"].str.len() > 0]
    return df.drop_duplicates(subset=["symbol"]).sort_values("symbol").reset_index(drop=True)


def fetch_sp500() -> pd.DataFrame:
    """Scrape the current S&P 500 constituent table from Wikipedia."""
    tables = pd.read_html(SP500_WIKI_URL)
    if not tables:
        raise RuntimeError(f"No tables found at {SP500_WIKI_URL}")
    # The constituents table is the first one carrying a Symbol column.
    for raw in tables:
        cols = {str(c).strip().lower() for c in raw.columns}
        if "symbol" in cols:
            return _normalise_wiki_frame(raw)
    raise RuntimeError("No table with a 'Symbol' column found on the Wikipedia page")


def load_sp500(refresh: bool = False) -> pd.DataFrame:
    """S&P 500 constituents, from cache unless `refresh=True`.

    Columns: symbol, security, gics_sector, gics_sub_industry, cik.

    A refresh that fails falls back to the cache when one exists — a
    scheduled run should not die because Wikipedia is briefly unreachable.
    """
    if not refresh and SP500_CACHE.exists():
        return pd.read_parquet(SP500_CACHE)

    try:
        df = fetch_sp500()
    except Exception as e:  # noqa: BLE001
        if SP500_CACHE.exists():
            log.warning("S&P 500 refresh failed (%s) — using cache at %s", e, SP500_CACHE)
            return pd.read_parquet(SP500_CACHE)
        raise RuntimeError(
            f"Could not fetch the S&P 500 list ({e}) and no cache exists at "
            f"{SP500_CACHE}. Check your connection, or use --universe etfs, "
            f"which needs no download."
        ) from e

    df.to_parquet(SP500_CACHE, index=False)
    log.info("Cached %d S&P 500 constituents to %s", len(df), SP500_CACHE)
    return df


def symbols(refresh: bool = False) -> list[str]:
    """S&P 500 tickers, Yahoo-normalised."""
    return load_sp500(refresh=refresh)["symbol"].tolist()


@lru_cache(maxsize=1)
def etf_symbols() -> list[str]:
    """The curated ETF universe — needs no network access."""
    return list(ETFS)


def etf_table() -> pd.DataFrame:
    """ETF universe as a frame: symbol, name, bucket."""
    return pd.DataFrame(
        [{"symbol": s, "name": n, "bucket": b} for s, (n, b) in ETFS.items()]
    )


def all_symbols(refresh: bool = False) -> list[str]:
    """ETFs first, then S&P 500 names not already covered."""
    etfs = etf_symbols()
    seen = set(etfs)
    return etfs + [s for s in symbols(refresh=refresh) if s not in seen]


def by_sector(refresh: bool = False) -> dict[str, list[str]]:
    """{GICS sector: [symbols]} for the S&P 500.

    Used by `features.correlations.sector_correlation`; ETFs are excluded
    because they have no GICS sector of their own.
    """
    df = load_sp500(refresh=refresh)
    return {
        str(sector): group["symbol"].tolist()
        for sector, group in df.groupby("gics_sector", sort=True)
    }


def sector_of(symbol: str) -> str | None:
    """GICS sector for one ticker, or None if it is not an S&P 500 name."""
    df = load_sp500()
    hit = df.loc[df["symbol"] == normalise_symbol(symbol), "gics_sector"]
    return str(hit.iloc[0]) if len(hit) else None
