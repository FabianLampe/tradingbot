"""Walk-forward backtest for the predictor.

Walk-forward = chronological train/test slicing. We never train on the
future. Concretely:

  1. Sort rows by date.
  2. Pick a series of "test windows" (e.g. 1 month each).
  3. For each test window: train on everything strictly *before* it,
     predict on the window, score the predictions, slide forward.

This is the only honest way to backtest a market model — random
train/test splits are useless because the same calendar day appears in
both train and test as different stocks.

Strategy implemented here: **Top-N long-only** (and optionally short).
Each rebalance day:
  - Predict scores for all symbols available that day.
  - Long top-N highest scores (equal-weight), short bottom-N (optional).
  - Hold for `horizon_days`, then close.
  - Subtract `cost_bps` per side (entry + exit).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.model.predictor import Predictor


@dataclass
class BacktestConfig:
    horizon_days: int = 5
    rebalance_days: int = 5             # equal to horizon: non-overlapping trades
    top_n_long: int = 10
    top_n_short: int = 0                # 0 = long-only
    cost_bps: float = 5.0               # per side (entry + exit each pay this)
    initial_capital: float = 100_000.0
    train_window_years: int | None = None   # None = expanding window
    test_window_days: int = 21          # ~1 trading month
    leverage: float = 1.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series              # capital over time
    trades: pd.DataFrame                 # one row per closed trade
    daily_returns: pd.Series             # portfolio simple-return per day
    metrics: dict                        # Sharpe, MDD, win rate, etc.


def _annualised_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    return float(np.sqrt(252) * daily_returns.mean() / daily_returns.std())


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def walk_forward(
    dataset: pd.DataFrame,
    predictor_factory,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a walk-forward backtest.

    Args:
      dataset:           full supervised dataset from build_dataset.build_dataset()
      predictor_factory: callable returning a fresh Predictor (we re-train each window)
      config:            BacktestConfig

    Returns BacktestResult.
    """
    df = dataset.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    # carve test windows
    first_test_idx = max(252 * 2, int(len(all_dates) * 0.4))   # ≥2y of training
    test_starts = all_dates[first_test_idx::config.test_window_days]

    trades: list[dict] = []
    daily_pnl: dict[pd.Timestamp, float] = {}

    for win_start in tqdm(test_starts, desc="Walk-forward"):
        win_end_idx = min(
            len(all_dates) - 1,
            list(all_dates).index(win_start) + config.test_window_days,
        )
        win_end = all_dates[win_end_idx]

        train_df = df[df["date"] < win_start].copy()
        test_df = df[(df["date"] >= win_start) & (df["date"] < win_end)].copy()

        if config.train_window_years is not None:
            cutoff = win_start - pd.Timedelta(days=365 * config.train_window_years)
            train_df = train_df[train_df["date"] >= cutoff]

        if train_df.empty or test_df.empty:
            continue

        # validation slice = last 10% of training, chronologically
        val_split = train_df["date"].quantile(0.9)
        val_df = train_df[train_df["date"] >= val_split]
        train_df = train_df[train_df["date"] < val_split]

        predictor = predictor_factory()
        predictor.train(train_df, df_val=val_df)

        # rebalance days within the window
        rebal_dates = sorted(test_df["date"].unique())[:: config.rebalance_days]
        for rebal_date in rebal_dates:
            day = test_df[test_df["date"] == rebal_date]
            if day.empty:
                continue
            scores = predictor.predict_score(day)
            day = day.assign(score=scores)

            longs = day.nlargest(config.top_n_long, "score")
            shorts = day.nsmallest(config.top_n_short, "score") if config.top_n_short else pd.DataFrame()

            n_pos = len(longs) + len(shorts)
            if n_pos == 0:
                continue
            weight = (config.leverage / n_pos)
            cost = (config.cost_bps / 10_000) * 2  # entry + exit
            for _, row in longs.iterrows():
                trades.append({
                    "entry_date": rebal_date, "symbol": row["symbol"],
                    "side": "long", "score": row["score"],
                    "fwd_return": row["fwd_return"],
                    "weight": weight,
                    "pnl": weight * (np.exp(row["fwd_return"]) - 1 - cost),
                })
            for _, row in shorts.iterrows():
                trades.append({
                    "entry_date": rebal_date, "symbol": row["symbol"],
                    "side": "short", "score": row["score"],
                    "fwd_return": row["fwd_return"],
                    "weight": weight,
                    "pnl": weight * (-(np.exp(row["fwd_return"]) - 1) - cost),
                })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=trades_df,
            daily_returns=pd.Series(dtype=float),
            metrics={"error": "no trades produced"},
        )

    # PnL hits the *exit* date conceptually; we simplify by attributing to entry date
    daily_returns = (
        trades_df.groupby("entry_date")["pnl"].sum().reindex(all_dates, fill_value=0.0)
    )
    equity = config.initial_capital * (1 + daily_returns).cumprod()

    metrics = {
        "n_trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl"] > 0).mean()),
        "mean_pnl_per_trade": float(trades_df["pnl"].mean()),
        "total_return": float(equity.iloc[-1] / config.initial_capital - 1),
        "annualised_return": float(
            (equity.iloc[-1] / config.initial_capital) ** (252 / max(len(daily_returns), 1)) - 1
        ),
        "sharpe": _annualised_sharpe(daily_returns),
        "max_drawdown": _max_drawdown(equity),
        "calmar": (
            (equity.iloc[-1] / config.initial_capital - 1)
            / abs(_max_drawdown(equity))
            if _max_drawdown(equity) < 0 else float("inf")
        ),
    }
    return BacktestResult(
        equity_curve=equity,
        trades=trades_df,
        daily_returns=daily_returns,
        metrics=metrics,
    )
