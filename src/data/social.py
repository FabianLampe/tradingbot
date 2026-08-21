"""Reddit ingestion — retail discussion as a sentiment source.

Complements `data.news` (what publishers report) and `data.rss_news` (what
moves the whole market) with what retail investors are actually talking
about. `runtime.daily.step_refresh_reddit` persists the output to
`data/news/_reddit.parquet`.

Subreddits come from `config/whitelist.yaml: reddit`. That list is curated
for discussion quality — wallstreetbets, pennystocks and the crypto-pump
subs are deliberately excluded, because their sentiment tracks coordinated
promotion rather than information.

Schema is FinBERT-ready (`headline` + `summary`, same as `data.news`):

    news_id        str
    datetime       datetime64[ns, UTC]   post creation time
    source         str                   "r/<subreddit>"
    subreddit      str
    headline       str                   post title
    summary        str                   self-text (truncated)
    url            str                   the link the post points at
    permalink      str                   the reddit thread itself
    score          int                   net upvotes
    num_comments   int
    upvote_ratio   float
    author         str
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, Sequence

import pandas as pd
from tqdm.auto import tqdm

import config
from src.data._common import clean_text, empty_frame, finalize, to_utc

log = logging.getLogger("trading_bot.data.social")

SOCIAL_COLUMNS: tuple[str, ...] = (
    "news_id", "datetime", "source", "subreddit", "headline", "summary",
    "url", "permalink", "score", "num_comments", "upvote_ratio", "author",
)

VALID_TIME_FILTERS = ("hour", "day", "week", "month", "year", "all")


def empty_social_frame() -> pd.DataFrame:
    """Correctly-typed empty frame with the canonical social schema."""
    return empty_frame(SOCIAL_COLUMNS)


def whitelisted_subreddits() -> list[str]:
    """Curated subreddit list from `whitelist.yaml: reddit`."""
    entries = config.load_whitelist().get("reddit") or []
    return [str(s).strip().lstrip("r/") for s in entries if str(s).strip()]


@lru_cache(maxsize=1)
def reddit_client():
    """Cached read-only PRAW client, or None when credentials are missing."""
    if not (config.REDDIT_CLIENT_ID and config.REDDIT_CLIENT_SECRET):
        return None
    import praw  # lazy import — praw is only needed for this one source

    client = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        check_for_async=False,
    )
    client.read_only = True
    return client


def _row(post, subreddit: str) -> dict | None:
    ts = to_utc(getattr(post, "created_utc", None))
    if ts is None:
        return None
    author = getattr(post, "author", None)
    return {
        "news_id": f"reddit:{post.id}",
        "datetime": ts,
        "source": f"r/{subreddit}",
        "subreddit": subreddit,
        "headline": clean_text(getattr(post, "title", ""), max_chars=512),
        "summary": clean_text(getattr(post, "selftext", ""), max_chars=2000),
        "url": str(getattr(post, "url", "") or ""),
        "permalink": f"https://www.reddit.com{getattr(post, 'permalink', '')}",
        "score": int(getattr(post, "score", 0) or 0),
        "num_comments": int(getattr(post, "num_comments", 0) or 0),
        "upvote_ratio": float(getattr(post, "upvote_ratio", 0.0) or 0.0),
        "author": str(author.name) if author is not None else "[deleted]",
    }


def fetch_subreddit(
    subreddit: str,
    time_filter: str = "day",
    limit: int = 100,
    sort: str = "top",
) -> pd.DataFrame:
    """Fetch posts from one subreddit. Empty frame on any error.

    `sort="top"` with `time_filter="day"` is the default because it filters
    out the long tail of zero-engagement posts before they ever reach
    FinBERT; `sort="new"` ignores `time_filter` (Reddit's API does).
    """
    client = reddit_client()
    if client is None:
        log.warning("Reddit credentials missing — set REDDIT_CLIENT_ID/SECRET in .env")
        return empty_social_frame()
    if time_filter not in VALID_TIME_FILTERS:
        raise ValueError(f"time_filter must be one of {VALID_TIME_FILTERS}, got {time_filter!r}")

    try:
        sub = client.subreddit(subreddit)
        if sort == "new":
            listing = sub.new(limit=limit)
        elif sort == "hot":
            listing = sub.hot(limit=limit)
        else:
            listing = sub.top(time_filter=time_filter, limit=limit)
        rows = [r for r in (_row(p, subreddit) for p in listing) if r is not None]
    except Exception as e:  # noqa: BLE001 — private sub, ban, rate limit, outage
        log.warning("[r/%s] fetch failed: %s", subreddit, e)
        return empty_social_frame()

    return finalize(rows, SOCIAL_COLUMNS)


def fetch_whitelisted(
    time_filter: str = "day",
    limit_per_sub: int = 100,
    sort: str = "top",
    subreddits: Sequence[str] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch every whitelisted subreddit into one frame.

    A failing subreddit is skipped, not fatal — the daily run should still
    get sentiment from the other five.
    """
    subs = list(subreddits) if subreddits is not None else whitelisted_subreddits()
    if not subs:
        log.warning("No subreddits configured — check `reddit:` in whitelist.yaml")
        return empty_social_frame()

    iterator: Iterable[str] = subs
    if show_progress:
        iterator = tqdm(subs, desc="Reddit", unit="sub")

    frames = [
        fetch_subreddit(s, time_filter=time_filter, limit=limit_per_sub, sort=sort)
        for s in iterator
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty_social_frame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["news_id"], keep="last")
    log.info("Reddit: %d posts from %d subreddits", len(combined), len(subs))
    return combined.sort_values("datetime").reset_index(drop=True)
