"""CLI: check the model for drift right now (one-shot)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import drift


def main():
    rep = drift.assess_drift()
    print(drift.format_drift_alert_de(rep))
    print()
    print(f"  rolling_winrate:    {rep.rolling_winrate*100:>6.2f}%")
    print(f"  baseline_winrate:   {rep.baseline_winrate*100:>6.2f}%")
    print(f"  rolling_avg_return: {rep.rolling_avg_return*100:>+6.2f}%")
    print(f"  baseline_avg_return:{rep.baseline_avg_return*100:>+6.2f}%")
    sys.exit(2 if rep.drift_detected else 0)


if __name__ == "__main__":
    main()
