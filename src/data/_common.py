"""Shared helpers for the news ingestion modules.

Kept in one place so `news`, `rss_news` and `social` produce byte-for-byte
compatible frames — downstream (`features.sentiment`, `runtime.premarket`,
`storage.db`) treats them interchangeably.
"""
from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from typing import Callable, Iterable, TypeVar

import pandas as pd

T = TypeVar("T")

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
# Stripping a tag leaves a space in front of the punctuation it hugged
# ("unchanged</b>." -> "unchanged ."). FinBERT tokenises that as its own
# token, so glue it back on.
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]])")

# Canonical datetime dtype for every frame this package emits. Pandas infers
# second resolution from parsed strings but nanosecond from unix ints; pinning
# it keeps `pd.concat` and parquet round-trips from producing mixed dtypes.
UTC_DTYPE = "datetime64[ns, UTC]"

# Suffixes stripped when reducing a domain/outlet name to a comparison key.
_TLDS = {"com", "net", "org", "co", "uk", "io", "gov", "edu", "de", "info", "news"}


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic short id — same article always yields the same id.

    Dedupe across refreshes depends on this: re-fetching an overlapping
    window must not create duplicate rows.
    """
    raw = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(raw.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def clean_text(value: object, max_chars: int = 2000) -> str:
    """Strip HTML, collapse whitespace, truncate. Never returns None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = _WS.sub(" ", _HTML_TAG.sub(" ", str(value)))
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()
    return text[:max_chars]


def outlet_key(name: object) -> str:
    """Reduce an outlet name or URL to a comparison key.

    'https://www.benzinga.com/x' -> 'benzinga', 'Benzinga' -> 'benzinga',
    'finance.yahoo.com' -> 'yahoo'. Lets the whitelist list domains while
    news APIs report human-readable outlet names.
    """
    s = str(name or "").strip().lower()
    s = re.sub(r"^[a-z]+://", "", s).split("/")[0]
    s = s.removeprefix("www.")
    parts = [p for p in s.split(".") if p and p not in _TLDS]
    return (parts[-1] if parts else s).replace(" ", "").replace("-", "")


def to_utc(value: object) -> pd.Timestamp | None:
    """Coerce unix seconds / struct_time / str / datetime to tz-aware UTC."""
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            if value <= 0:
                return None
            return pd.Timestamp(int(value), unit="s", tz="UTC")
        if isinstance(value, time.struct_time):
            return pd.Timestamp(datetime(*value[:6], tzinfo=timezone.utc))
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if ts is pd.NaT:
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def empty_frame(columns: Iterable[str], datetime_col: str = "datetime") -> pd.DataFrame:
    """Typed empty frame — keeps `pd.concat` from producing object dtypes."""
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    if datetime_col in df.columns:
        df[datetime_col] = pd.Series(dtype=UTC_DTYPE)
    return df


def finalize(
    rows: list[dict],
    columns: Iterable[str],
    datetime_col: str = "datetime",
    id_col: str = "news_id",
) -> pd.DataFrame:
    """Rows -> tidy frame: fixed column order, UTC datetimes, deduped, sorted."""
    columns = list(columns)
    if not rows:
        return empty_frame(columns, datetime_col)
    df = pd.DataFrame(rows)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    df[datetime_col] = pd.to_datetime(
        df[datetime_col], utc=True, errors="coerce"
    ).astype(UTC_DTYPE)
    df = df.dropna(subset=[datetime_col])
    if id_col in df.columns:
        df = df.drop_duplicates(subset=[id_col], keep="last")
    return df.sort_values(datetime_col).reset_index(drop=True)


class RateLimiter:
    """Blocking min-interval limiter (Finnhub free tier: 60 calls/min)."""

    def __init__(self, calls_per_minute: int):
        self.min_interval = 60.0 / max(1, calls_per_minute)
        self._last = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last
        if self._last and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


def with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 2.0,
    on_error: Callable[[int, Exception], None] | None = None,
    give_up_on: tuple[type[BaseException], ...] = (),
) -> T:
    """Call `fn`, retrying transient failures with exponential backoff.

    `give_up_on` names exception types that are known to be permanent
    (a rejected API key, an unknown series id) — those are re-raised
    immediately instead of being retried into a long, pointless wait.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except give_up_on:
            raise
        except Exception as e:  # noqa: BLE001 — provider SDKs raise freely
            last = e
            if on_error is not None:
                on_error(attempt, e)
            if attempt == attempts:
                break
            time.sleep(base_delay * (2 ** (attempt - 1)))
    raise last  # type: ignore[misc]
