"""Returns and panel construction.

A "panel" is a wide DataFrame with date on the index and one column per
ticker. Most analytics (correlations, beta, factor models) want this
shape rather than long-format. We build it lazily from the per-ticker
Parquet files in `data/prices/`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.storage import db


def build_close_panel(
    symbols: list[str] | None = None,
    column: str = "adj_close",
) -> pd.DataFrame:
    """Wide DataFrame: index=date, columns=symbol, values=`column`.

    Missing data (delistings, IPOs after panel start) appear as NaN —
    leave them NaN; downstream code uses `min_periods` to handle this.
    """
    prices = db.read_all_prices(symbols)
    if not prices:
        raise RuntimeError(
            "No cached prices. Run notebooks/02_download_sp500_history.ipynb first."
        )
    closes = pd.DataFrame({sym: df[column] for sym, df in prices.items()})
    closes.index = pd.to_datetime(closes.index)
    return closes.sort_index()


def log_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Log returns. Use these for ML features — they are additive over time
    and roughly symmetric, which most models prefer over simple returns."""
    return np.log(close_panel / close_panel.shift(1))


def simple_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns. Use these for performance reporting
    (a 5% gain reads naturally; a 0.0488 log-return does not)."""
    return close_panel.pct_change()


def forward_return(
    close_panel: pd.DataFrame,
    horizon_days: int,
    method: str = "log",
) -> pd.DataFrame:
    """Return realised over the *next* `horizon_days` business days.

    This is the supervised-learning target. Beware look-ahead bias:
    when training, only ever use rows where `index < today - horizon`.
    """
    if method == "log":
        r = np.log(close_panel.shift(-horizon_days) / close_panel)
    elif method == "simple":
        r = close_panel.shift(-horizon_days) / close_panel - 1.0
    else:
        raise ValueError(f"Unknown method: {method}")
    return r
