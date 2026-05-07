"""CLI: bulk-ingest SEC EDGAR filings into the RAG knowledge base."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import universe
from src.knowledge import edgar


def main():
    p = argparse.ArgumentParser(description="Ingest SEC EDGAR filings into RAG")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="sp500")
    p.add_argument("--limit", type=int, default=2,
                   help="Filings per form per ticker (be polite to SEC)")
    p.add_argument("--email", default="research@example.com",
                   help="Required by SEC for the User-Agent header")
    args = p.parse_args()

    if args.universe == "etfs":
        # ETFs don't file 10-K/Q in the way operating companies do — skip
        print("ETFs don't file standard 10-K/Q. Use --universe sp500.")
        return
    elif args.universe == "sp500":
        symbols = universe.symbols()
    else:
        symbols = [s for s in universe.all_symbols() if s not in universe.etf_symbols()]

    n = edgar.ingest_universe(symbols, limit_per_form=args.limit, user_agent_email=args.email)
    print(f"\nIngested {n} document chunks into RAG.")


if __name__ == "__main__":
    main()
