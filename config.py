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
NEWS_DIR = DATA_DIR / "news"
MACRO_DIR = DATA_DIR / "macro"
META_DIR = DATA_DIR / "meta"

for d in (PRICES_DIR, NEWS_DIR, MACRO_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- API keys (None if not set — modules check before calling) ----
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY") or None
FRED_API_KEY = os.getenv("FRED_API_KEY") or None
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or None

# ---- Defaults ----
# History depth for initial S&P 500 download. 15y is enough to span
# 2008 crisis, 2020 COVID crash, and 2022 rate-hike cycle.
DEFAULT_HISTORY_YEARS = 15
DEFAULT_BENCHMARK = "^GSPC"  # S&P 500 index itself
