"""CLI: pre-market run (~14:30 dt. time)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import daily


def main():
    p = argparse.ArgumentParser(description="Pre-market overnight-news adjustment")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="etfs")
    p.add_argument("--no-telegram", action="store_true")
    args = p.parse_args()

    summary = daily.run_premarket(universe_size=args.universe)

    if not args.no_telegram:
        try:
            from app.telegram_bot import push_today
            push_today()
        except Exception as e:  # noqa: BLE001
            print(f"[telegram] push failed: {e}")

    print("Pre-market summary:", summary)


if __name__ == "__main__":
    main()
