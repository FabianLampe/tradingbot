"""Project-wide configuration.

Loads .env once and exposes constants used by every module.
Importing this module has no side effects beyond reading the .env file.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# ---- Storage paths ----
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data")).resolve()
PRICES_DIR = DATA_DIR / "prices"
INTRADAY_DIR = DATA_DIR / "prices_intraday"
NEWS_DIR = DATA_DIR / "news"
MACRO_DIR = DATA_DIR / "macro"
META_DIR = DATA_DIR / "meta"
MODELS_DIR = DATA_DIR / "models"
JOURNAL_DIR = DATA_DIR / "journal"
PAPER_DIR = DATA_DIR / "paper"
CONFIG_DIR = PROJECT_ROOT / "config"

for d in (PRICES_DIR, INTRADAY_DIR, NEWS_DIR, MACRO_DIR, META_DIR,
          MODELS_DIR, JOURNAL_DIR, PAPER_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_whitelist() -> dict:
    """Load curated source whitelist (Twitter/Reddit/News)."""
    import yaml
    path = CONFIG_DIR / "whitelist.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

# ---- API keys (None if not set — modules check before calling) ----
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or None
FRED_API_KEY = os.getenv("FRED_API_KEY") or None
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or None
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID") or None
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET") or None
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT") or "trading-bot/0.1"

# ---- Defaults ----
# History depth for initial S&P 500 download. 15y is enough to span
# 2008 crisis, 2020 COVID crash, and 2022 rate-hike cycle.
DEFAULT_HISTORY_YEARS = 15
DEFAULT_BENCHMARK = "^GSPC"  # S&P 500 index itself
