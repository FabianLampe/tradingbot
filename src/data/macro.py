"""Macro panel from FRED (Federal Reserve Economic Data).

Produces a daily, date-indexed panel that `features.build_dataset` broadcasts
onto every price row, and that `app.dashboard` plots.

**Publication lag is applied on purpose.** FRED indexes an observation by its
*reference* period, not its release date: January CPI sits on 1 January but is
not published until mid-February. Forward-filling it as-is would leak six
weeks of future information into every backtest row — exactly the failure
mode `build_dataset` warns about ("each row uses only information available at
its date"). Each series therefore declares a `release_lag_days`, and we shift
its dates forward by that before reindexing onto the daily calendar.

Daily market series (VIX, Treasury yields, spreads) carry a lag of 0: the
end-of-day pipeline runs after the close, when that day's value is published.

Without `FRED_API_KEY` every function returns empty — the daily run logs a
warning and continues without macro features rather than failing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import pandas as pd

import config
from src.data._common import with_retries

log = logging.getLogger("trading_bot.data.macro")


class _PermanentFredError(RuntimeError):
    """A FRED rejection that retrying cannot fix (bad key, unknown series)."""


@dataclass(frozen=True)
class MacroSeries:
    """One FRED series and how far behind its reference date it publishes."""
    series_id: str
    name: str
    release_lag_days: int
    note: str = ""


# Daily series first, then the lower-frequency ones.
SERIES: tuple[MacroSeries, ...] = (
    # --- risk / rates (daily, same-day publication) ---
    MacroSeries("VIXCLS", "vix", 0, "CBOE volatility index — the fear gauge"),
    MacroSeries("DFF", "fed_funds_rate", 0, "Effective federal funds rate"),
    MacroSeries("DGS2", "treasury_2y", 0),
    MacroSeries("DGS10", "treasury_10y", 0),
    MacroSeries("T10Y2Y", "yield_curve_10y_2y", 0, "Classic recession signal when negative"),
    MacroSeries("T10Y3M", "yield_curve_10y_3m", 0),
    MacroSeries("BAMLH0A0HYM2", "high_yield_spread", 0, "Credit stress — widens before equity drawdowns"),
    MacroSeries("DTWEXBGS", "dollar_index", 0, "Broad trade-weighted USD"),
    MacroSeries("DCOILWTICO", "oil_wti", 0),
    # --- macro fundamentals (monthly, published weeks after the fact) ---
    MacroSeries("CPIAUCSL", "cpi", 30, "CPI for month M lands mid-M+1"),
    MacroSeries("UNRATE", "unemployment_rate", 15, "Jobs report, first Friday after month end"),
    MacroSeries("INDPRO", "industrial_production", 20),
    MacroSeries("UMCSENT", "consumer_sentiment", 15),
)

PANEL_COLUMNS: tuple[str, ...] = tuple(s.name for s in SERIES)


def empty_panel() -> pd.DataFrame:
    """Correctly-typed empty panel."""
    df = pd.DataFrame({c: pd.Series(dtype="float64") for c in PANEL_COLUMNS})
    df.index = pd.DatetimeIndex([], name="date")
    return df


@lru_cache(maxsize=1)
def _fred():
    """Cached FRED client, or None when no key is configured."""
    if not config.FRED_API_KEY:
        return None
    from fredapi import Fred  # lazy import

    return Fred(api_key=config.FRED_API_KEY)


def fetch_series(
    series_id: str,
    start: str | date | None = "2005-01-01",
    end: str | date | None = None,
) -> pd.Series:
    """Fetch one raw FRED series, indexed by its **reference** date.

    No publication lag is applied here — this is the plotting/exploration
    surface. `fetch_panel` is the leakage-safe one.
    """
    client = _fred()
    if client is None:
        log.warning("No FRED_API_KEY configured — returning empty series")
        return pd.Series(dtype="float64", name=series_id)

    def _call():
        try:
            return client.get_series(
                series_id,
                observation_start=str(start) if start else None,
                observation_end=str(end) if end else None,
            )
        except ValueError as e:
            # fredapi raises ValueError for "Bad Request" — an unknown series
            # id or a rejected API key. Retrying that just burns time, and a
            # bad key means every series in the panel fails the same way.
            raise _PermanentFredError(str(e)) from e

    try:
        raw = with_retries(
            _call,
            attempts=3,
            on_error=lambda attempt, e: log.warning(
                "[%s] FRED attempt %d failed: %s", series_id, attempt, e
            ),
            give_up_on=(_PermanentFredError,),
        )
    except _PermanentFredError as e:
        log.error("[%s] FRED rejected the request: %s", series_id, e)
        return pd.Series(dtype="float64", name=series_id)
    except Exception as e:  # noqa: BLE001
        log.error("[%s] FRED fetch failed: %s", series_id, e)
        return pd.Series(dtype="float64", name=series_id)

    s = pd.Series(raw, dtype="float64")
    # `.as_unit("ns")` keeps this index joinable with the price index, which
    # is pinned the same way — pandas otherwise infers us or ns per source.
    s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize().as_unit("ns")
    s.index.name = "date"
    return s.dropna().sort_index().rename(series_id)


def fetch_panel(
    start: str | date | None = "2005-01-01",
    end: str | date | None = None,
    apply_release_lag: bool = True,
    series: tuple[MacroSeries, ...] = SERIES,
) -> pd.DataFrame:
    """Daily macro panel, forward-filled and leakage-safe.

    Columns are the `name` fields of `SERIES`; the index is a business-day
    calendar. Each series is shifted by its publication lag (unless
    `apply_release_lag=False`) before being forward-filled, so a row dated D
    only contains values that were actually public on D.

    A series that fails to download is skipped with a warning — a partial
    panel is more useful to the daily run than no panel at all.
    """
    if _fred() is None:
        log.warning("No FRED_API_KEY configured — macro panel will be empty")
        return empty_panel()

    columns: dict[str, pd.Series] = {}
    for spec in series:
        s = fetch_series(spec.series_id, start=start, end=end)
        if s.empty:
            log.warning("[%s] no data — column '%s' omitted", spec.series_id, spec.name)
            continue
        if apply_release_lag and spec.release_lag_days:
            s.index = s.index + pd.Timedelta(days=spec.release_lag_days)
        columns[spec.name] = s.rename(spec.name)

    if not columns:
        log.error("FRED returned no usable series — check the API key")
        return empty_panel()

    panel = pd.concat(columns.values(), axis=1).sort_index()

    # Reindex onto business days and forward-fill: a weekly/monthly reading
    # stays "the latest known value" until the next release supersedes it.
    calendar = pd.bdate_range(panel.index.min(), panel.index.max(), name="date")
    panel = panel.reindex(panel.index.union(calendar)).ffill().reindex(calendar)

    if end is not None:
        panel = panel[panel.index <= pd.Timestamp(end)]

    panel.index.name = "date"
    panel = panel.astype("float64")
    log.info("Macro panel: %d rows x %d series", len(panel), panel.shape[1])
    return panel
