"""SQLite-backed Trade Journal.

Every recommendation the bot ever makes gets written here, *with all its
inputs*. After the holding horizon is up, we record what actually
happened. This is the foundation for Phase 3:

  - SHAP attribution can be re-computed from stored inputs
  - LLM post-mortems read winning/losing trades from here
  - Drift monitoring queries running win-rate

Schema (deliberately denormalised for read-speed):

  predictions(
      id, created_at, asof_date, symbol, score,
      action,                  -- 'long' | 'short' | 'flat'
      horizon_days, top_features_json, sentiment_inputs_json,
      macro_snapshot_json, model_version
  )
  outcomes(
      prediction_id PK, evaluated_at, realised_return,
      success_bool, notes
  )
  postmortems(
      prediction_id PK, written_at, llm_model, analysis_md
  )
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import JOURNAL_DIR

DEFAULT_DB = JOURNAL_DIR / "journal.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    asof_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    score REAL NOT NULL,
    action TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    top_features_json TEXT,
    sentiment_inputs_json TEXT,
    macro_snapshot_json TEXT,
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_date_sym ON predictions(asof_date, symbol);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id INTEGER PRIMARY KEY,
    evaluated_at TEXT NOT NULL,
    realised_return REAL NOT NULL,
    success_bool INTEGER NOT NULL,
    notes TEXT,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);

CREATE TABLE IF NOT EXISTS postmortems (
    prediction_id INTEGER PRIMARY KEY,
    written_at TEXT NOT NULL,
    llm_model TEXT NOT NULL,
    analysis_md TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
);
"""


@dataclass
class PredictionRecord:
    asof_date: str            # YYYY-MM-DD
    symbol: str
    score: float              # signed [-1, 1]
    action: str               # 'long' | 'short' | 'flat'
    horizon_days: int
    top_features: dict        # {feature_name: shap_value or raw_value}
    sentiment_inputs: dict    # {n_articles, mean_score, top_headlines: [...]}
    macro_snapshot: dict      # {vix, fed_funds, ...}
    model_version: str = "predictor-v1"


@contextmanager
def _conn(db_path: Path = DEFAULT_DB):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def log_prediction(rec: PredictionRecord, db_path: Path = DEFAULT_DB) -> int:
    with _conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO predictions
               (created_at, asof_date, symbol, score, action, horizon_days,
                top_features_json, sentiment_inputs_json, macro_snapshot_json,
                model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                rec.asof_date, rec.symbol, rec.score, rec.action, rec.horizon_days,
                json.dumps(rec.top_features),
                json.dumps(rec.sentiment_inputs),
                json.dumps(rec.macro_snapshot),
                rec.model_version,
            ),
        )
        return cur.lastrowid


def log_outcome(
    prediction_id: int,
    realised_return: float,
    success: bool,
    notes: str | None = None,
    db_path: Path = DEFAULT_DB,
) -> None:
    with _conn(db_path) as c:
        c.execute(
            """INSERT OR REPLACE INTO outcomes
               (prediction_id, evaluated_at, realised_return, success_bool, notes)
               VALUES (?, ?, ?, ?, ?)""",
            (
                prediction_id,
                datetime.now(timezone.utc).isoformat(),
                realised_return,
                int(bool(success)),
                notes,
            ),
        )


def log_postmortem(
    prediction_id: int,
    llm_model: str,
    analysis_md: str,
    db_path: Path = DEFAULT_DB,
) -> None:
    with _conn(db_path) as c:
        c.execute(
            """INSERT OR REPLACE INTO postmortems
               (prediction_id, written_at, llm_model, analysis_md)
               VALUES (?, ?, ?, ?)""",
            (
                prediction_id,
                datetime.now(timezone.utc).isoformat(),
                llm_model,
                analysis_md,
            ),
        )


# ---------- queries ----------

def recent_predictions(days: int = 30, db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    with _conn(db_path) as c:
        return pd.read_sql_query(
            """SELECT * FROM predictions
               WHERE asof_date >= date('now', ?)
               ORDER BY asof_date DESC, score DESC""",
            c, params=(f"-{days} days",),
        )


def predictions_pending_outcome(
    horizon_days: int,
    db_path: Path = DEFAULT_DB,
) -> pd.DataFrame:
    """Predictions that should have a realised outcome by now but don't."""
    with _conn(db_path) as c:
        return pd.read_sql_query(
            """SELECT p.* FROM predictions p
               LEFT JOIN outcomes o ON o.prediction_id = p.id
               WHERE o.prediction_id IS NULL
                 AND date(p.asof_date, '+' || p.horizon_days || ' days') <= date('now')""",
            c,
        )


def failed_trades(db_path: Path = DEFAULT_DB, limit: int = 50) -> pd.DataFrame:
    """Worst recent trades — Phase 3 post-mortem feeds on these."""
    with _conn(db_path) as c:
        return pd.read_sql_query(
            """SELECT p.*, o.realised_return, o.evaluated_at FROM predictions p
               JOIN outcomes o ON o.prediction_id = p.id
               WHERE o.success_bool = 0
               ORDER BY o.realised_return ASC
               LIMIT ?""",
            c, params=(limit,),
        )


def journal_summary(db_path: Path = DEFAULT_DB) -> dict:
    with _conn(db_path) as c:
        cur = c.cursor()
        n_pred = cur.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        n_out = cur.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        win_rate = cur.execute(
            "SELECT AVG(success_bool) FROM outcomes"
        ).fetchone()[0]
        avg_ret = cur.execute(
            "SELECT AVG(realised_return) FROM outcomes"
        ).fetchone()[0]
    return {
        "n_predictions": n_pred,
        "n_evaluated": n_out,
        "win_rate": win_rate or 0.0,
        "mean_realised_return": avg_ret or 0.0,
    }
