"""CLI: generate LLM post-mortems for recent failed trades.

For every losing trade in the journal that doesn't already have a
post-mortem, this asks the configured LLM to analyse what went wrong
and writes the result back to the journal.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import postmortem as pm_mod
from src.model import journal as journal_mod


def main():
    p = argparse.ArgumentParser(description="Generate LLM post-mortems for failed trades")
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    failures = journal_mod.failed_trades(limit=args.limit)
    if failures.empty:
        print("Keine Fehltrades zu analysieren.")
        return

    # filter out those that already have a postmortem
    db_path = journal_mod.DEFAULT_DB
    with sqlite3.connect(str(db_path)) as c:
        existing = set(r[0] for r in c.execute("SELECT prediction_id FROM postmortems"))

    todo = failures[~failures["id"].isin(existing)]
    print(f"{len(todo)} Postmortems zu generieren ...")

    for _, row in todo.iterrows():
        req = pm_mod.PostmortemRequest(
            symbol=row["symbol"],
            asof_date=row["asof_date"],
            action=row["action"],
            score=float(row["score"]),
            horizon_days=int(row["horizon_days"]),
            realised_return=float(row["realised_return"]),
            top_features=json.loads(row.get("top_features_json") or "[]"),
            sentiment_inputs=json.loads(row.get("sentiment_inputs_json") or "{}"),
            macro_snapshot=json.loads(row.get("macro_snapshot_json") or "{}"),
        )
        try:
            text, model_id = pm_mod.write_postmortem(req)
            journal_mod.log_postmortem(int(row["id"]), llm_model=model_id, analysis_md=text)
            print(f"  ✓ {row['symbol']} {row['asof_date']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {row['symbol']} {row['asof_date']}: {e}")


if __name__ == "__main__":
    main()
