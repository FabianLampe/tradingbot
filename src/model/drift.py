"""Drift detection — is the model getting worse?

Reads the trade journal, computes rolling performance, and flags when
the recent stretch is materially worse than the longer-term baseline.

Two complementary signals:
  1. **Performance drift** — rolling 30-trade win-rate dropped >10pp
     vs. all-time win-rate.
  2. **Score-distribution drift** — model is producing unusually
     extreme or unusually flat scores compared to what it used to.

When drift fires, the daily orchestrator triggers a full retrain
*and* sends a Telegram alert.

Conscious choice: the threshold values are deliberately simple. Tune
them in production based on your tolerance — too sensitive = needless
retrains, too lax = quietly degrading model.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.model import journal as journal_mod


@dataclass
class DriftReport:
    rolling_winrate: float
    baseline_winrate: float
    winrate_drop_pp: float       # percentage points
    rolling_avg_return: float
    baseline_avg_return: float
    return_drop_pp: float
    drift_detected: bool
    reason: str


def assess_drift(
    rolling_n: int = 30,
    winrate_drop_threshold: float = 0.10,
    return_drop_threshold: float = 0.005,
) -> DriftReport:
    """Compare last `rolling_n` evaluated trades to the all-time baseline.

    Args:
      rolling_n:               window size for the recent slice
      winrate_drop_threshold:  trigger if rolling - baseline win-rate < -X (default 10pp)
      return_drop_threshold:   trigger if rolling - baseline mean-return < -X (default 0.5pp)
    """
    import sqlite3
    with sqlite3.connect(str(journal_mod.DEFAULT_DB)) as c:
        df = pd.read_sql_query(
            """SELECT p.id, p.asof_date, p.action,
                      o.realised_return, o.success_bool, o.evaluated_at
               FROM predictions p
               JOIN outcomes o ON o.prediction_id = p.id
               ORDER BY o.evaluated_at""",
            c,
        )
    if df.empty or len(df) < max(rolling_n + 10, 20):
        return DriftReport(
            rolling_winrate=0.0, baseline_winrate=0.0, winrate_drop_pp=0.0,
            rolling_avg_return=0.0, baseline_avg_return=0.0, return_drop_pp=0.0,
            drift_detected=False,
            reason=f"Not enough data yet ({len(df)} evaluated trades).",
        )

    recent = df.tail(rolling_n)
    baseline = df.iloc[: -rolling_n] if len(df) > rolling_n else df

    rolling_wr = float(recent["success_bool"].mean())
    base_wr = float(baseline["success_bool"].mean())
    wr_drop = rolling_wr - base_wr

    rolling_ret = float(recent["realised_return"].mean())
    base_ret = float(baseline["realised_return"].mean())
    ret_drop = rolling_ret - base_ret

    drift = (wr_drop < -winrate_drop_threshold) or (ret_drop < -return_drop_threshold)
    reason_parts = []
    if wr_drop < -winrate_drop_threshold:
        reason_parts.append(
            f"Win-Rate gefallen um {-wr_drop*100:.1f}pp "
            f"({base_wr*100:.1f}% → {rolling_wr*100:.1f}%)"
        )
    if ret_drop < -return_drop_threshold:
        reason_parts.append(
            f"Mean-Return gefallen um {-ret_drop*100:.2f}pp "
            f"({base_ret*100:+.2f}% → {rolling_ret*100:+.2f}%)"
        )
    reason = "; ".join(reason_parts) if reason_parts else "Keine Drift erkannt."

    return DriftReport(
        rolling_winrate=rolling_wr,
        baseline_winrate=base_wr,
        winrate_drop_pp=wr_drop,
        rolling_avg_return=rolling_ret,
        baseline_avg_return=base_ret,
        return_drop_pp=ret_drop,
        drift_detected=drift,
        reason=reason,
    )


def format_drift_alert_de(rep: DriftReport) -> str:
    """Markdown-formatted alert for Telegram."""
    if not rep.drift_detected:
        return f"✅ *Drift-Check OK* — {rep.reason}"
    return (
        "🚨 *DRIFT ERKANNT*\n\n"
        f"{rep.reason}\n\n"
        f"Rolling Win-Rate: *{rep.rolling_winrate*100:.1f}%*\n"
        f"Baseline Win-Rate: *{rep.baseline_winrate*100:.1f}%*\n"
        f"Rolling Ø Return: *{rep.rolling_avg_return*100:+.2f}%*\n\n"
        "_Komplett-Retrain wird beim nächsten EOD-Lauf erzwungen._"
    )
