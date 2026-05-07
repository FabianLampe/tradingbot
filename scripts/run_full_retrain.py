"""CLI: force a full from-scratch retrain.

Use after schema changes, after long maintenance gaps, or when drift
fired and you want to retrain manually instead of waiting for the next
EOD job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import daily


def main():
    p = argparse.ArgumentParser(description="Force full retrain (no warm-start)")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="etfs")
    args = p.parse_args()
    summary = daily.run_daily_eod(universe_size=args.universe, force_full_retrain=True)
    print("Full retrain summary:", summary)


if __name__ == "__main__":
    main()
