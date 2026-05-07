"""CLI entry point for the daily End-of-Day pipeline.

Usage:
    python scripts/run_daily.py                # default: ETFs only (fast)
    python scripts/run_daily.py --universe sp500
    python scripts/run_daily.py --universe all
    python scripts/run_daily.py --no-telegram  # skip telegram push
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import daily


def main():
    p = argparse.ArgumentParser(description="Trading bot daily run")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="etfs")
    p.add_argument("--no-telegram", action="store_true", help="skip Telegram push")
    args = p.parse_args()

    summary = daily.run_daily_eod(universe_size=args.universe)

    if not args.no_telegram and summary.get("recommendations", 0) > 0:
        try:
            from app.telegram_bot import push_today
            push_today()
            print("[telegram] push sent.")
        except Exception as e:  # noqa: BLE001
            print(f"[telegram] push failed: {e}")

    print("\nFinal summary:", summary)


if __name__ == "__main__":
    main()
