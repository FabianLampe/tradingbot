"""CLI: walk-forward backtest. Saves equity curve and trade list to data/journal/."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from src.data import prices, universe
from src.features import build_dataset
from src.model import backtest as bt_mod
from src.model import costs as cost_mod
from src.model.predictor import Predictor
from src.storage import db


def main():
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--universe", choices=["etfs", "sp500", "all"], default="etfs")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--top-n-long", type=int, default=10)
    p.add_argument("--top-n-short", type=int, default=0)
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="Flat per-side cost; ignored unless --cost-preset=flat")
    p.add_argument("--cost-preset", default="neobroker_1eur",
                   choices=sorted(cost_mod.PRESETS) + ["flat"],
                   help="Realistic cost model. 'flat' restores the old "
                        "--cost-bps behaviour.")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--initial", type=float, default=100_000.0)
    p.add_argument("--no-cross-sectional", action="store_true",
                   help="Use raw feature levels instead of per-date ranks "
                        "(lets the model trade market regime, not selection)")
    p.add_argument("--cross-sectional-method", choices=["rank", "zscore"], default="rank")
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip the benchmark download and report absolute returns only")
    args = p.parse_args()

    if args.universe == "etfs":
        symbols = universe.etf_symbols()
    elif args.universe == "sp500":
        symbols = universe.symbols()
    else:
        symbols = universe.all_symbols()

    if not args.no_benchmark and not db.price_path(bt_mod.BENCHMARK_SYMBOL).exists():
        print(f"Benchmark {config.DEFAULT_BENCHMARK} not cached — downloading ...")
        prices.download_benchmark()

    print(f"Building dataset for {len(symbols)} symbols, horizon {args.horizon}d ...")
    dataset = build_dataset.build_dataset(
        symbols,
        horizon_days=args.horizon,
        cross_sectional=not args.no_cross_sectional,
        cross_sectional_method=args.cross_sectional_method,
    )
    if dataset.empty:
        print("Empty dataset — make sure prices are downloaded (Notebook 02).")
        return

    feature_cols = build_dataset.feature_columns(dataset)
    model = None if args.cost_preset == "flat" else cost_mod.get_preset(args.cost_preset)
    n_pos = args.top_n_long + args.top_n_short
    if model is not None:
        print()
        print(cost_mod.format_viability(model, n_positions=max(n_pos, 1),
                                        holding_days=args.horizon))
        print()

    config_bt = bt_mod.BacktestConfig(
        horizon_days=args.horizon,
        rebalance_days=args.horizon,
        top_n_long=args.top_n_long,
        top_n_short=args.top_n_short,
        cost_bps=args.cost_bps,
        cost_model=model,
        leverage=args.leverage,
        initial_capital=args.initial,
    )

    transform = build_dataset.feature_transform_of(dataset)
    print(f"Feature scaling: {transform}")

    def factory():
        return Predictor(feature_cols=feature_cols, horizon_days=args.horizon,
                         feature_transform=transform)

    print("Running walk-forward (this can take a few minutes) ...")
    result = bt_mod.walk_forward(dataset, factory, config_bt)

    eq_path = config.JOURNAL_DIR / "backtest_equity.parquet"
    tr_path = config.JOURNAL_DIR / "backtest_trades.parquet"
    pd.Series(result.equity_curve, name="equity").to_frame().to_parquet(eq_path)
    result.trades.to_parquet(tr_path, index=False)

    def _show(title, keys):
        shown = [(k, result.metrics[k]) for k in keys if k in result.metrics]
        if not shown:
            return
        print(f"\n=== {title} ===")
        for k, v in shown:
            print(f"  {k:28} {v:>10.4f}" if isinstance(v, float) else f"  {k:28} {v}")

    _show("STRATEGY (absolute)", [
        "n_trades", "n_rebalances", "traded_years", "win_rate",
        "mean_pnl_per_trade", "total_return", "annualised_return",
        "sharpe", "max_drawdown", "calmar",
    ])
    _show("COSTS", [
        "cost_model", "avg_position_size", "avg_round_trip_cost",
        "gross_total_return", "gross_annualised_return", "cost_drag_annualised",
    ])
    if result.metrics.get("ruined"):
        print(f"\n  [!] RUINED on {result.metrics['ruin_date']} after "
              f"{result.metrics['survived_years']:.1f} years.")
        print(f"      {result.metrics['ruin_note']}")
    _show("VS BENCHMARK", [
        "benchmark_symbol", "benchmark_periods", "benchmark_total_return",
        "benchmark_annualised_return", "benchmark_sharpe",
        "excess_total_return", "excess_annualised_return",
        "information_ratio", "alpha_annualised", "beta_vs_benchmark",
        "hit_rate_vs_benchmark",
    ])
    if err := result.metrics.get("benchmark_error"):
        print(f"\n  [!] {err}")
    else:
        ir = result.metrics.get("information_ratio")
        if ir is not None:
            verdict = ("beats buy-and-hold" if ir > 0.3
                       else "no meaningful edge over buy-and-hold")
            print(f"\n  -> Information ratio {ir:.2f}: {verdict}.")
            print("     The absolute return above is mostly market exposure "
                  "unless this number is clearly positive.")

    print(f"\nEquity curve -> {eq_path}")
    print(f"Trades       -> {tr_path}")


if __name__ == "__main__":
    main()
