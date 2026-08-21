"""CLI: backfill news into the parquet cache.

Company news is written per ticker per year (`data/news/{SYM}_{YEAR}.parquet`),
market RSS to `data/news/_rss_market.parquet`, Reddit to `data/news/_reddit.parquet` —
exactly the layout `runtime.daily` refreshes incrementally and
`features.build_dataset` reads. Re-running is idempotent: articles dedupe
on `news_id`.

Examples:
    python scripts/ingest_news.py --symbols AAPL,MSFT --days-back 30
    python scripts/ingest_news.py --symbols-file config/etfs.txt --score
    python scripts/ingest_news.py --rss --reddit
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from src.data import news as news_mod
from src.data import rss_news, social
from src.storage import db


def _read_symbols(args) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols += [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.symbols_file:
        text = Path(args.symbols_file).read_text(encoding="utf-8")
        symbols += [line.strip().upper() for line in text.splitlines()
                    if line.strip() and not line.startswith("#")]
    return list(dict.fromkeys(symbols))


def ingest_company_news(symbols: list[str], days_back: int, score: bool) -> int:
    """Fetch, optionally score, and merge into the per-year parquet files."""
    bulk = news_mod.fetch_bulk_company_news(symbols, days_back=days_back)

    scorer = None
    if score:
        from src.features.sentiment import FinBERTScorer
        scorer = FinBERTScorer()   # heavy — load once, reuse for all symbols

    total = 0
    for sym, df in bulk.items():
        if df.empty:
            continue
        if scorer is not None:
            from src.features.sentiment import score_news_dataframe
            df = score_news_dataframe(df, scorer=scorer)

        # A window can straddle a year boundary — split before writing.
        by_year: dict[int, list[pd.DataFrame]] = defaultdict(list)
        for year, chunk in df.groupby(df["datetime"].dt.year):
            by_year[int(year)].append(chunk)

        for year, chunks in by_year.items():
            incoming = pd.concat(chunks, ignore_index=True)
            path = db.news_path(sym, year)
            existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
            db.write_news(sym, year, news_mod.merge_news(existing, incoming))
            total += len(incoming)
        print(f"  {sym:<6} {len(df):>5} articles")
    return total


def main():
    p = argparse.ArgumentParser(description="Backfill news into the parquet cache")
    p.add_argument("--symbols", help="Comma-separated tickers, e.g. AAPL,MSFT")
    p.add_argument("--symbols-file", help="File with one ticker per line")
    p.add_argument("--days-back", type=int, default=30,
                   help="History window in days (Finnhub free tier: ~1 year)")
    p.add_argument("--score", action="store_true",
                   help="Run FinBERT before writing (needs torch + transformers)")
    p.add_argument("--rss", action="store_true", help="Also refresh market RSS")
    p.add_argument("--reddit", action="store_true", help="Also refresh Reddit")
    args = p.parse_args()

    symbols = _read_symbols(args)
    if not symbols and not (args.rss or args.reddit):
        p.error("give --symbols/--symbols-file, and/or --rss / --reddit")

    if symbols:
        print(f"Company news: {len(symbols)} tickers, {args.days_back} days back")
        total = ingest_company_news(symbols, args.days_back, args.score)
        print(f"-> {total} articles cached in {config.NEWS_DIR}")

    if args.rss:
        df = rss_news.fetch_all_whitelisted()
        if df.empty:
            print("RSS: nothing fetched")
        else:
            out = config.NEWS_DIR / "_rss_market.parquet"
            df.to_parquet(out, index=False)
            print(f"RSS: {len(df)} items -> {out}")

    if args.reddit:
        df = social.fetch_whitelisted(time_filter="day", limit_per_sub=100)
        if df.empty:
            print("Reddit: nothing fetched")
        else:
            out = config.NEWS_DIR / "_reddit.parquet"
            df.to_parquet(out, index=False)
            print(f"Reddit: {len(df)} posts -> {out}")


if __name__ == "__main__":
    main()
