"""Market-wide RSS news ingestion.

Company news (`data.news`) tells you what happened to *one* ticker. This
module covers the other half: Reuters/Bloomberg/Fed/BLS headlines that move
*everything*. `features.cross_sentiment.market_sentiment_from_rss` turns the
output into the daily market-sentiment feature.

Which outlets get pulled is driven by `config/whitelist.yaml`:

  - `news_outlets`           -> the domains we trust (the actual selection)
  - `news_outlets_blacklist` -> domains skipped even if listed above
  - `rss_feeds`              -> optional {domain: [feed-url, ...]} override

Only domains present in `news_outlets` are fetched. Feed URLs come from
`rss_feeds` when given, otherwise from `DEFAULT_FEEDS` below — so a moved
or dead feed is a YAML edit, not a code change.

Schema (compatible with `data.news`, `title` instead of `headline` because
that is what `cross_sentiment` renames):

    news_id   str
    datetime  datetime64[ns, UTC]
    source    str   outlet domain, e.g. "reuters.com"
    title     str
    summary   str
    url       str
    feed      str   the feed URL the item came from
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm

import config
from src.data._common import (
    clean_text,
    empty_frame,
    finalize,
    outlet_key,
    stable_id,
    to_utc,
    with_retries,
)

log = logging.getLogger("trading_bot.data.rss")

RSS_COLUMNS: tuple[str, ...] = (
    "news_id", "datetime", "source", "title", "summary", "url", "feed",
)

# Several outlets 403 the default feedparser user-agent.
USER_AGENT = "trading-bot/0.1 (+research; contact: local)"

# Fallback feed map, used for any whitelisted domain not overridden in
# `whitelist.yaml: rss_feeds`. Outlets without a usable public feed
# (bloomberg.com, wsj.com full text, …) are intentionally absent — they are
# still whitelisted for *source attribution*, just not RSS-pullable.
DEFAULT_FEEDS: dict[str, tuple[str, ...]] = {
    "reuters.com": ("https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",),
    "bloomberg.com": ("https://feeds.bloomberg.com/markets/news.rss",),
    "ft.com": ("https://www.ft.com/rss/home",),
    "wsj.com": (
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.a.dj.com/rss/RSSWSJD.xml",
    ),
    "economist.com": ("https://www.economist.com/finance-and-economics/rss.xml",),
    "marketwatch.com": ("https://feeds.content.dowjones.io/public/rss/mw_topstories",),
    "cnbc.com": (
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",   # top news
        "https://www.cnbc.com/id/20910258/device/rss/rss.html",    # markets
    ),
    "finance.yahoo.com": ("https://finance.yahoo.com/news/rssindex",),
    "seekingalpha.com": ("https://seekingalpha.com/market_currents.xml",),
    "barrons.com": ("https://feeds.a.dj.com/rss/RSSBarronsMarketsMain.xml",),
    "sec.gov": ("https://www.sec.gov/news/pressreleases.rss",),
    "federalreserve.gov": ("https://www.federalreserve.gov/feeds/press_all.xml",),
    "bls.gov": ("https://www.bls.gov/feed/bls_latest.rss",),
    "bea.gov": ("https://www.bea.gov/rss.xml",),
}


def empty_rss_frame() -> pd.DataFrame:
    """Correctly-typed empty frame with the canonical RSS schema."""
    return empty_frame(RSS_COLUMNS)


def feed_map(whitelist: Mapping | None = None) -> dict[str, list[str]]:
    """Resolve {domain: [feed urls]} for every whitelisted, non-blacklisted outlet."""
    wl = dict(whitelist) if whitelist is not None else config.load_whitelist()
    outlets = wl.get("news_outlets") or []
    blocked = {outlet_key(b) for b in (wl.get("news_outlets_blacklist") or []) if b}
    overrides = wl.get("rss_feeds") or {}

    resolved: dict[str, list[str]] = {}
    for domain in outlets:
        domain = str(domain).strip().lower()
        if not domain or outlet_key(domain) in blocked:
            continue
        urls = overrides.get(domain, DEFAULT_FEEDS.get(domain, ()))
        if isinstance(urls, str):
            urls = [urls]
        urls = [str(u) for u in urls if u]
        if urls:
            resolved[domain] = urls
        else:
            log.debug("No RSS feed known for whitelisted outlet %s", domain)
    return resolved


def fetch_feed(
    url: str,
    source: str | None = None,
    max_items: int | None = 200,
    timeout: int = 15,
) -> pd.DataFrame:
    """Fetch and parse one RSS/Atom feed. Returns an empty frame on failure."""
    import feedparser  # lazy: keeps `import config`-only callers light

    source = source or outlet_key(url)
    try:
        raw = with_retries(
            lambda: requests.get(
                url, timeout=timeout, headers={"User-Agent": USER_AGENT}
            ).content,
            attempts=2,
            on_error=lambda attempt, e: log.warning(
                "[%s] feed attempt %d failed: %s", source, attempt, e
            ),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] feed unreachable (%s): %s", source, url, e)
        return empty_rss_frame()

    parsed = feedparser.parse(raw)
    entries = parsed.entries or []
    if max_items is not None:
        entries = entries[:max_items]

    rows: list[dict] = []
    for entry in entries:
        ts = to_utc(
            entry.get("published_parsed")
            or entry.get("updated_parsed")
            or entry.get("published")
            or entry.get("updated")
        )
        if ts is None:
            # An undated item cannot be aligned to a trading day — skip it
            # rather than silently stamping it "now" and skewing the panel.
            continue
        link = str(entry.get("link") or "")
        rows.append({
            "news_id": stable_id("rss", entry.get("id") or link, entry.get("title")),
            "datetime": ts,
            "source": source,
            "title": clean_text(entry.get("title"), max_chars=512),
            "summary": clean_text(entry.get("summary") or entry.get("description")),
            "url": link,
            "feed": url,
        })
    if not rows:
        log.debug("[%s] no dated entries in %s", source, url)
    return finalize(rows, RSS_COLUMNS)


def fetch_all_whitelisted(
    max_items_per_feed: int | None = 200,
    timeout: int = 15,
    show_progress: bool = True,
    whitelist: Mapping | None = None,
) -> pd.DataFrame:
    """Pull every whitelisted feed into one frame.

    Called by `runtime.daily.step_refresh_rss`, which persists the result to
    `data/news/_rss_market.parquet`. Dead feeds are logged and skipped — a
    single 403 must not cost us the whole market-sentiment feature.
    """
    feeds = feed_map(whitelist)
    if not feeds:
        log.warning("No RSS feeds resolved — check news_outlets in whitelist.yaml")
        return empty_rss_frame()

    targets: list[tuple[str, str]] = [
        (domain, url) for domain, urls in feeds.items() for url in urls
    ]
    iterator: Iterable[tuple[str, str]] = targets
    if show_progress:
        iterator = tqdm(targets, desc="RSS", unit="feed")

    frames = [
        fetch_feed(url, source=domain, max_items=max_items_per_feed, timeout=timeout)
        for domain, url in iterator
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_rss_frame()

    combined = pd.concat(frames, ignore_index=True)
    # Wire stories are syndicated verbatim across outlets — dedupe on the
    # headline too, else one Reuters item gets counted five times.
    combined = combined.drop_duplicates(subset=["news_id"], keep="last")
    combined = combined.drop_duplicates(subset=["title", "source"], keep="last")
    log.info("RSS: %d items from %d feeds", len(combined), len(targets))
    return combined.sort_values("datetime").reset_index(drop=True)
