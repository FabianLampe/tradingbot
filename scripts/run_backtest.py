"""CLI: walk-forward backtest. Saves equity curve and trade list to data/journal/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from src.data import universe
from src.features import build_dataset
from src.model import backtest as bt_mod
from src.model.predictor import Predictor


def main():
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="etfs")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--top-n-long", type=int, default=10)
    p.add_argument("--top-n-short", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--initial", type=float, default=100_000.0)
    args = p.parse_args()

    if args.universe == "etfs":
        symbols = universe.etf_symbols()
    elif args.universe == "sp500":
        symbols = universe.symbols()
    else:
        symbols = universe.all_symbols()

    print(f"Building dataset for {len(symbols)} symbols, horizon {args.horizon}d ...")
    dataset = build_dataset.build_dataset(symbols, horizon_days=args.horizon)
    if dataset.empty:
        print("Empty dataset — make sure prices are downloaded (Notebook 02).")
        return

    feature_cols = build_dataset.feature_columns(dataset)
    config_bt = bt_mod.BacktestConfig(
        horizon_days=args.horizon,
        rebalance_days=args.horizon,
        top_n_long=args.top_n_long,
        top_n_short=args.top_n_short,
        cost_bps=args.cost_bps,
        leverage=args.leverage,
        initial_capital=args.initial,
    )

    def factory():
        return Predictor(feature_cols=feature_cols, horizon_days=args.horizon)

    print("Running walk-forward (this can take a few minutes) ...")
    result = bt_mod.walk_forward(dataset, factory, config_bt)

    eq_path = config.JOURNAL_DIR / "backtest_equity.parquet"
    tr_path = config.JOURNAL_DIR / "backtest_trades.parquet"
    pd.Series(result.equity_curve, name="equity").to_frame().to_parquet(eq_path)
    result.trades.to_parquet(tr_path, index=False)

    print("\n=== BACKTEST METRICS ===")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            print(f"  {k:24} {v:>10.4f}")
        else:
            print(f"  {k:24} {v}")
    print(f"\nEquity curve -> {eq_path}")
    print(f"Trades       -> {tr_path}")


if __name__ == "__main__":
    main()
