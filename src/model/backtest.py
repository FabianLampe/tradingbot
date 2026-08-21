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

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config as project_config
from src.model.costs import CostModel
from src.model.predictor import Predictor
from src.storage import db

log = logging.getLogger("trading_bot.backtest")

# Notebook 02 and scripts/ingest_prices cache the benchmark under this name —
# "^GSPC" is not a legal filename stem, so the underscore form is the convention.
BENCHMARK_SYMBOL = "_GSPC"


@dataclass
class BacktestConfig:
    horizon_days: int = 5
    rebalance_days: int = 5             # equal to horizon: non-overlapping trades
    top_n_long: int = 10
    top_n_short: int = 0                # 0 = long-only
    cost_bps: float = 5.0               # per side; used only when cost_model is None
    initial_capital: float = 100_000.0
    train_window_years: int | None = None   # None = expanding window
    test_window_days: int = 21          # ~1 trading month
    leverage: float = 1.0
    benchmark_symbol: str = BENCHMARK_SYMBOL
    ruin_fraction: float = 0.01
    """Stop the run once equity falls below this fraction of the starting
    capital. A wiped-out account is a terminal state, not a small number to
    keep compounding: with a fixed per-order fee, a shrinking account pays an
    ever-larger *percentage* per trade, so continuing produces a death spiral
    and meaningless infinities instead of the finding, which is 'ruined'."""

    cost_model: CostModel | None = None
    """Realistic costs (spread + slippage + commission). When set, position
    notional is tracked through the run so a fixed per-order fee shows up as
    the percentage it actually is — which is what decides whether a strategy
    works at a given account size. Leave None for the flat `cost_bps`
    behaviour the earlier results were produced with."""

    def resolved_cost_model(self) -> CostModel:
        return self.cost_model or CostModel(
            spread_bps=self.cost_bps, slippage_bps=0.0, name="flat_from_cost_bps",
        )


@dataclass
class BacktestResult:
    equity_curve: pd.Series              # capital over time
    trades: pd.DataFrame                 # one row per closed trade
    daily_returns: pd.Series             # portfolio simple-return per day
    metrics: dict                        # Sharpe, MDD, win rate, alpha, IR, ...
    benchmark_returns: pd.Series | None = None   # aligned to the holding periods


def _annualised_sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    """Sharpe at a risk-free rate of 0, annualised from the series' own frequency."""
    if returns.empty or returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std())


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def _annualise(total_return: float, years: float) -> float:
    """Geometric annualisation over the actual elapsed time."""
    if years <= 0:
        return 0.0
    growth = 1.0 + total_return
    if growth <= 0:
        # Total loss. Raising a non-positive base to a fractional power returns
        # a complex number in Python, so short-circuit: nothing left is -100%,
        # however long it took to get there.
        return -1.0
    return float(growth ** (1.0 / years) - 1.0)


def load_benchmark(symbol: str = BENCHMARK_SYMBOL) -> pd.Series | None:
    """Adjusted-close series for the benchmark, or None if it is not cached."""
    try:
        px = db.read_prices(symbol)
    except FileNotFoundError:
        log.warning(
            "Benchmark %s not cached — skipping benchmark-relative metrics. "
            "Cache it with: python -c \"from src.data import prices; "
            "prices.download_benchmark()\"  (downloads %s).",
            symbol, project_config.DEFAULT_BENCHMARK,
        )
        return None
    if px.empty or "adj_close" not in px.columns:
        return None
    s = px["adj_close"].copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def benchmark_holding_returns(
    benchmark: pd.Series,
    entry_dates: pd.DatetimeIndex,
    horizon_days: int,
) -> pd.Series:
    """Benchmark return over each trade's holding window.

    The portfolio books one `horizon_days` return per entry date. Comparing
    that against *daily* benchmark returns would be comparing different
    horizons, so we measure the benchmark over exactly the same windows:
    entry close -> close `horizon_days` business days later.
    """
    idx = benchmark.index
    out = {}
    for entry in entry_dates:
        try:
            i = idx.get_indexer([entry], method="bfill")[0]
        except Exception:  # noqa: BLE001
            continue
        j = i + horizon_days
        if i < 0 or j >= len(idx):
            continue
        out[entry] = float(benchmark.iloc[j] / benchmark.iloc[i] - 1.0)
    return pd.Series(out, name="benchmark").sort_index()


def _relative_metrics(
    portfolio: pd.Series,
    benchmark: pd.Series,
    periods_per_year: float,
) -> dict:
    """Alpha, beta, information ratio — all on aligned holding-period returns.

    Risk-free rate is taken as 0. At current short rates that flatters both
    the strategy and the benchmark by roughly the same amount, so the
    *relative* numbers (alpha, IR) stay honest; the absolute Sharpe does not.

    Read `alpha_annualised` together with `beta_vs_benchmark`, not on its own.
    Alpha is return *net of the market exposure actually taken*, so a
    market-neutral strategy (beta ~ 0) earning 6% shows alpha 6% even in a year
    the index did 12% — correct, but not "it beat the market". Conversely a
    2x-levered index tracker shows large excess return and alpha ~ 0, which is
    the honest verdict: leverage is not skill. `information_ratio` is the one
    number that does not need this caveat; treat it as the headline.
    """
    joined = pd.concat([portfolio.rename("p"), benchmark.rename("b")], axis=1).dropna()
    if len(joined) < 3:
        return {"benchmark_error": f"only {len(joined)} overlapping periods"}

    p, b = joined["p"], joined["b"]
    var_b = float(b.var())
    beta = float(p.cov(b) / var_b) if var_b > 0 else float("nan")

    excess = p - b                      # active return vs a 1x benchmark position
    bench_total = float((1.0 + b).prod() - 1.0)
    port_total = float((1.0 + p).prod() - 1.0)
    years = len(joined) / periods_per_year

    metrics = {
        "benchmark_symbol": benchmark.attrs.get("symbol", BENCHMARK_SYMBOL),
        "benchmark_periods": int(len(joined)),
        "benchmark_total_return": bench_total,
        "benchmark_annualised_return": _annualise(bench_total, years),
        "benchmark_sharpe": _annualised_sharpe(b, periods_per_year),
        "excess_total_return": port_total - bench_total,
        "excess_annualised_return": (
            _annualise(port_total, years) - _annualise(bench_total, years)
        ),
        # Information ratio: the headline number. Mean active return divided by
        # its volatility. Below ~0.3 the strategy is not beating buy-and-hold
        # by enough to justify the turnover and the risk of being wrong.
        "information_ratio": _annualised_sharpe(excess, periods_per_year),
        "beta_vs_benchmark": beta,
    }
    if not np.isnan(beta):
        # Jensen's alpha: the part of the return not explained by market exposure.
        alpha_per_period = float(p.mean() - beta * b.mean())
        metrics["alpha_annualised"] = alpha_per_period * periods_per_year
        metrics["hit_rate_vs_benchmark"] = float((excess > 0).mean())
    return metrics


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
    cost_model = config.resolved_cost_model()
    equity_now = config.initial_capital
    ruin_level = config.initial_capital * config.ruin_fraction
    ruined_at: pd.Timestamp | None = None

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
            # Position size in currency, so a fixed commission converts to the
            # right percentage. This is why equity is tracked through the loop
            # rather than reconstructed afterwards.
            notional = equity_now * weight
            period_return = 0.0

            for _, row in longs.iterrows():
                gross = np.exp(row["fwd_return"]) - 1
                cost_frac = cost_model.round_trip_fraction(
                    notional, config.horizon_days, "long")
                pnl = weight * (gross - cost_frac)
                period_return += pnl
                trades.append({
                    "entry_date": rebal_date, "symbol": row["symbol"],
                    "side": "long", "score": row["score"],
                    "fwd_return": row["fwd_return"],
                    "weight": weight, "notional": notional,
                    "gross_pnl": weight * gross,
                    "cost": weight * cost_frac,
                    "pnl": pnl,
                })
            for _, row in shorts.iterrows():
                gross = -(np.exp(row["fwd_return"]) - 1)
                cost_frac = cost_model.round_trip_fraction(
                    notional, config.horizon_days, "short")
                pnl = weight * (gross - cost_frac)
                period_return += pnl
                trades.append({
                    "entry_date": rebal_date, "symbol": row["symbol"],
                    "side": "short", "score": row["score"],
                    "fwd_return": row["fwd_return"],
                    "weight": weight, "notional": notional,
                    "gross_pnl": weight * gross,
                    "cost": weight * cost_frac,
                    "pnl": pnl,
                })

            # You cannot lose more than the capital committed to the period.
            period_return = max(period_return, -1.0)
            # Compound so the next rebalance sizes off the updated account.
            equity_now *= (1 + period_return)

            if equity_now <= ruin_level:
                ruined_at = pd.Timestamp(rebal_date)
                log.warning(
                    "Account ruined on %s: equity %.2f fell below %.2f (%.0f%% of "
                    "the starting %.2f). Stopping the run.",
                    ruined_at.date(), equity_now, ruin_level,
                    config.ruin_fraction * 100, config.initial_capital,
                )
                break
        if ruined_at is not None:
            break

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return BacktestResult(
            equity_curve=pd.Series(dtype=float),
            trades=trades_df,
            daily_returns=pd.Series(dtype=float),
            metrics={"error": "no trades produced"},
        )

    # One return per rebalance, each covering `horizon_days`. This — not the
    # zero-padded daily series — is the strategy's true return frequency, and
    # it is what the benchmark gets aligned to.
    #
    # Clipped at -100% for the same reason the simulation loop clips: without
    # leverage you cannot lose more than the capital committed, and letting a
    # period past -1 through would drive the equity curve negative and make
    # every downstream metric meaningless.
    period_returns = (
        trades_df.groupby("entry_date")["pnl"].sum().clip(lower=-1.0).sort_index()
    )
    period_returns.index = pd.DatetimeIndex(period_returns.index)

    # PnL hits the *exit* date conceptually; we simplify by attributing to entry date
    daily_returns = period_returns.reindex(all_dates, fill_value=0.0)
    equity = config.initial_capital * (1 + daily_returns).cumprod()

    # Annualise over the period actually traded. The dataset's first ~40% is
    # training-only warmup where equity sits flat at the initial capital;
    # including it would understate every annualised figure.
    first_trade, last_trade = period_returns.index.min(), period_returns.index.max()
    traded_years = max((last_trade - first_trade).days / 365.25, 1e-9)
    periods_per_year = len(period_returns) / traded_years
    total_return = float(equity.iloc[-1] / config.initial_capital - 1)
    mdd = _max_drawdown(equity)

    # What the strategy would have made with no trading frictions at all.
    # The gap between this and total_return is the whole cost question.
    gross_periods = trades_df.groupby("entry_date")["gross_pnl"].sum().sort_index()
    gross_total = float((1 + gross_periods).prod() - 1)

    metrics = {
        "n_trades": int(len(trades_df)),
        "n_rebalances": int(len(period_returns)),
        "traded_years": float(traded_years),
        "win_rate": float((trades_df["pnl"] > 0).mean()),
        "mean_pnl_per_trade": float(trades_df["pnl"].mean()),
        "total_return": total_return,
        "annualised_return": _annualise(total_return, traded_years),
        "sharpe": _annualised_sharpe(period_returns, periods_per_year),
        "max_drawdown": mdd,
        "calmar": total_return / abs(mdd) if mdd < 0 else float("inf"),
        # --- cost block ---
        "cost_model": cost_model.name,
        "gross_total_return": gross_total,
        "gross_annualised_return": _annualise(gross_total, traded_years),
        "cost_drag_annualised": (
            _annualise(gross_total, traded_years) - _annualise(total_return, traded_years)
        ),
        "avg_position_size": float(trades_df["notional"].mean()),
        "avg_round_trip_cost": float(
            (trades_df["cost"] / trades_df["weight"].replace(0, np.nan)).mean()
        ),
        "ruined": ruined_at is not None,
    }
    if ruined_at is not None:
        metrics["ruin_date"] = str(ruined_at.date())
        metrics["survived_years"] = float(
            (ruined_at - first_trade).days / 365.25
        )
        metrics["ruin_note"] = (
            f"Account fell below {config.ruin_fraction:.0%} of its starting capital "
            f"and the run stopped. Gross return was {gross_total:+.1%} over the same "
            f"trades — the strategy was killed by costs at this account size, not by "
            f"the signal."
        )

    bench_periods: pd.Series | None = None
    benchmark = load_benchmark(config.benchmark_symbol)
    if benchmark is not None:
        bench_periods = benchmark_holding_returns(
            benchmark, period_returns.index, config.horizon_days
        )
        bench_periods.attrs["symbol"] = config.benchmark_symbol
        metrics.update(_relative_metrics(period_returns, bench_periods, periods_per_year))
    else:
        metrics["benchmark_error"] = (
            f"{config.benchmark_symbol} not cached — absolute returns only. "
            "A long-only strategy in a bull market looks good on absolute "
            "numbers alone; cache the benchmark to see whether it is alpha."
        )

    return BacktestResult(
        equity_curve=equity,
        trades=trades_df,
        daily_returns=daily_returns,
        metrics=metrics,
        benchmark_returns=bench_periods,
    )
