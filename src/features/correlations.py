"""Correlation analytics across the S&P 500.

Three views matter for trading:
  1. **Static snapshot**: full N×N Pearson matrix on a window of returns.
     Used to find clusters and check sector cohesion.
  2. **Rolling pairwise**: how does corr(AAPL, MSFT) evolve over time?
     Used to detect regime changes (correlations spike to 1.0 in crashes).
  3. **Beta to benchmark**: rolling regression vs. ^GSPC.
     Tells us how much each stock "is" the market vs. idiosyncratic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import universe


def correlation_matrix(
    returns: pd.DataFrame,
    window: int | None = None,
    end: str | pd.Timestamp | None = None,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Pearson correlation matrix on a slice of the returns panel.

    Args:
        returns: wide DataFrame, columns=symbols, index=date.
        window: lookback in business days. If None, uses all rows up to ``end``.
        end:    last date to include (default = last available).
        min_periods: minimum overlapping observations per pair (default = window/2).
    """
    end = pd.Timestamp(end) if end is not None else returns.index.max()
    sliced = returns.loc[:end]
    if window is not None:
        sliced = sliced.tail(window)
    min_periods = min_periods or (len(sliced) // 2)
    return sliced.corr(min_periods=min_periods)


def rolling_pairwise(
    returns: pd.DataFrame,
    sym_a: str,
    sym_b: str,
    window: int = 60,
) -> pd.Series:
    """Rolling correlation between two symbols. 60-day window is the
    standard for "intermediate-term" correlation."""
    return returns[sym_a].rolling(window).corr(returns[sym_b])


def rolling_beta(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    window: int = 252,
) -> pd.DataFrame:
    """Rolling beta of every column in `returns` vs. a benchmark series.

    Beta is cov(stock, market) / var(market). 252 trading days = 1 year.
    """
    aligned = returns.join(benchmark.rename("__bench__"), how="inner")
    bench = aligned["__bench__"]
    stocks = aligned.drop(columns="__bench__")

    bench_var = bench.rolling(window).var()
    betas = pd.DataFrame(index=aligned.index, columns=stocks.columns, dtype=float)
    for sym in stocks.columns:
        cov = stocks[sym].rolling(window).cov(bench)
        betas[sym] = cov / bench_var
    return betas


def sector_correlation(
    returns: pd.DataFrame,
    window: int | None = 252,
) -> pd.DataFrame:
    """Mean intra-sector correlation, by GICS sector.

    A high number means stocks in that sector move as a herd (e.g. Energy
    during oil shocks). A low number means the sector is heterogeneous.
    """
    sectors = universe.by_sector()
    rows = []
    full_corr = correlation_matrix(returns, window=window)
    for sector, syms in sectors.items():
        present = [s for s in syms if s in full_corr.columns]
        if len(present) < 2:
            continue
        sub = full_corr.loc[present, present]
        # mean of strict upper triangle (exclude self-correlations of 1.0)
        mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        rows.append({
            "sector": sector,
            "n_stocks": len(present),
            "mean_corr": sub.values[mask].mean(),
            "median_corr": np.median(sub.values[mask]),
        })
    return pd.DataFrame(rows).sort_values("mean_corr", ascending=False)


def most_correlated_pairs(
    corr: pd.DataFrame,
    top_n: int = 20,
    exclude_same_sector: bool = False,
) -> pd.DataFrame:
    """Highest off-diagonal correlations.

    Set ``exclude_same_sector=True`` to find *cross-sector* couplings —
    these are often the interesting ones (e.g. an airline correlated to
    an oil refiner is informative; two oil majors are not).
    """
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs = corr.where(mask).stack().sort_values(ascending=False)

    if exclude_same_sector:
        sym_to_sector = dict(
            zip(universe.load_sp500()["symbol"], universe.load_sp500()["gics_sector"])
        )
        pairs = pairs[
            [sym_to_sector.get(a) != sym_to_sector.get(b) for a, b in pairs.index]
        ]

    out = pairs.head(top_n).reset_index()
    out.columns = ["symbol_a", "symbol_b", "correlation"]
    return out
