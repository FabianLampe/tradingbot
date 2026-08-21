"""CLI: run the event-driven strategy against a paper broker.

No real money moves. The point is to accumulate out-of-sample evidence: the
backtest saw data the model was built against, this sees data that did not
exist yet. When the two disagree, believe this one.

Typical use — a cron entry alongside the daily pipeline:

    python scripts/run_paper.py --symbols SPY,QQQ,XLK --deposit-monthly 100
    python scripts/run_paper.py --report          # where do I stand?
    python scripts/run_paper.py --replay 60       # simulate the last 60 days first

`--replay` is the honest starting point: it runs the same logic bar by bar
over cached history, so you see the cost structure and the trade frequency
before committing to months of waiting.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from src.data import intraday
from src.execution.paper import PaperBroker
from src.model import costs as cost_mod
from src.runtime.events import (
    EventConfig, exit_reason, find_events, format_funnel, signal_funnel,
    typical_move_from_bars,
)

log = logging.getLogger("trading_bot.paper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_feature_bars(
    symbols: list[str],
    interval: str,
    horizon_bars: int,
    scorer=None,
) -> pd.DataFrame:
    """Assemble (timestamp, symbol, score, sent_mean, n_articles, typical_move).

    Sentiment is joined from the cached news files by calendar day — hourly
    news alignment would need intraday timestamps matched bar by bar, which is
    the next step up in fidelity and is not what a first paper run needs.
    """
    from src.data import news as news_mod

    frames = []
    for sym in symbols:
        try:
            bars = intraday.read_intraday(sym, interval)
        except FileNotFoundError:
            log.warning("[%s] no %s bars cached — skipped", sym, interval)
            continue
        if bars.empty:
            continue

        df = pd.DataFrame(index=bars.index)
        df["symbol"] = sym
        df["close"] = bars["close"]
        df["typical_move"] = typical_move_from_bars(bars["close"], horizon_bars)

        # Momentum stand-in for the model score, so the runner works before a
        # predictor exists. Pass --use-model once one is trained.
        ret = np.log(bars["close"] / bars["close"].shift(1))
        z = ret.rolling(20, min_periods=5).mean() / ret.rolling(20, min_periods=5).std()
        df["score"] = np.tanh(z.fillna(0.0))

        df["date"] = df.index.tz_convert("UTC").normalize()
        sent = _daily_sentiment(sym)
        if sent is None:
            df["sent_mean"] = 0.0
            df["n_articles"] = 0
        else:
            df = df.join(sent, on="date")
            df[["sent_mean", "n_articles"]] = df[["sent_mean", "n_articles"]].fillna(0.0)
        frames.append(df.drop(columns=["date"]))

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames).reset_index().rename(columns={"index": "timestamp"})
    out["n_articles"] = out["n_articles"].astype(int)
    return out.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def _daily_sentiment(symbol: str) -> pd.DataFrame | None:
    files = sorted(config.NEWS_DIR.glob(f"{symbol}_*.parquet"))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if "score" not in df.columns or df.empty:
        return None
    df["date"] = pd.to_datetime(df["datetime"], utc=True).dt.normalize()
    return df.groupby("date").agg(sent_mean=("score", "mean"),
                                  n_articles=("score", "size"))


def replay(broker: PaperBroker, bars: pd.DataFrame, cfg: EventConfig, days: int) -> None:
    """Walk cached history bar by bar, trading the same logic the live run uses."""
    if bars.empty:
        print("No bars to replay — cache intraday data first.")
        return

    cutoff = bars["timestamp"].max() - pd.Timedelta(days=days)
    bars = bars[bars["timestamp"] >= cutoff]
    timestamps = sorted(bars["timestamp"].unique())
    print(f"Replaying {len(timestamps)} bars over {days} days, "
          f"{bars['symbol'].nunique()} symbols\n")

    open_at: dict[str, tuple[pd.Timestamp, float, int]] = {}
    for ts in timestamps:
        slice_ = bars[bars["timestamp"] == ts]
        marks = dict(zip(slice_["symbol"], slice_["close"]))
        equity = broker.equity(marks)

        # exits first — frees cash for the same bar's entries
        for sym in list(broker.positions):
            if sym not in marks or sym not in open_at:
                continue
            entry_ts, entry_px, direction = open_at[sym]
            held = int(((slice_["timestamp"].iloc[0] - entry_ts) /
                        pd.Timedelta(hours=1)))
            why = exit_reason(entry_px, marks[sym], direction, held, cfg)
            if why:
                broker.close_position(sym, marks[sym], ts.to_pydatetime(), note=why)
                open_at.pop(sym, None)

        if len(broker.positions) >= cfg.max_positions:
            continue

        notional = equity * cfg.position_fraction
        if notional <= 0:
            continue
        signals = find_events(slice_, cfg, broker.cost_model, notional,
                              holding_days=cfg.max_holding_bars / 6.5)
        for sig in signals:
            if len(broker.positions) >= cfg.max_positions or sig.symbol in broker.positions:
                continue
            price = marks.get(sig.symbol)
            if not price:
                continue
            qty = broker.target_quantity(price, min(notional, broker.cash * 0.95))
            if qty <= 0:
                continue
            fill = broker.market_order(sig.symbol, "buy" if sig.direction > 0 else "sell",
                                       qty, price, ts.to_pydatetime(), note=sig.reason)
            if fill:
                open_at[sig.symbol] = (ts, price, sig.direction)

        broker.record_equity(marks, ts.to_pydatetime())

    final = bars[bars["timestamp"] == timestamps[-1]]
    if not broker.fills:
        # Zero trades is a result, not a failure — but only if it says why.
        avg_notional = broker.deposited * cfg.position_fraction
        print(format_funnel(signal_funnel(
            bars, cfg, broker.cost_model, max(avg_notional, 1.0),
            holding_days=cfg.max_holding_bars / 6.5,
        )))
    report(broker, dict(zip(final["symbol"], final["close"])))


def report(broker: PaperBroker, marks: dict[str, float]) -> None:
    p = broker.performance(marks)
    print("\n=== PAPER PORTFOLIO ===")
    for k in ("equity", "cash", "positions_value", "deposited", "profit",
              "total_costs", "n_fills", "n_positions"):
        v = p[k]
        print(f"  {k:20} {v:>12,.2f}" if isinstance(v, float) else f"  {k:20} {v:>12}")
    print(f"  {'return_on_deposits':20} {p['return_on_deposits']:>11.2%}")
    print(f"  {'costs_pct_of_deposits':20} {p['costs_pct_of_deposits']:>11.2%}")

    if p["n_fills"]:
        gross = p["profit"] + p["total_costs"]
        print(f"\n  Before costs the strategy made {gross:,.2f}; "
              f"costs took {p['total_costs']:,.2f}.")
        if gross > 0 and p["profit"] <= 0:
            print("  -> The signal was profitable and the costs ate all of it. "
                  "Trade less often or in bigger size, not differently.")
        elif gross <= 0:
            print("  -> The signal itself lost money. Costs are not the problem here.")

    if broker.positions:
        print("\n  Open positions:")
        for s, pos in broker.positions.items():
            mark = marks.get(s, pos.avg_price)
            pnl = (mark - pos.avg_price) * pos.quantity
            print(f"    {s:<6} {pos.quantity:>10.4f} @ {pos.avg_price:>8.2f} "
                  f"mark {mark:>8.2f}  unreal. {pnl:>+8.2f}")


def main():
    p = argparse.ArgumentParser(description="Event-driven paper trading")
    p.add_argument("--symbols", default="SPY,QQQ,XLK")
    p.add_argument("--interval", default=intraday.DEFAULT_INTERVAL,
                   choices=intraday.supported_intervals())
    p.add_argument("--refresh", action="store_true", help="Download fresh bars first")
    p.add_argument("--days-back", type=int, default=None)
    p.add_argument("--replay", type=int, metavar="DAYS",
                   help="Simulate the last N days of cached bars")
    p.add_argument("--report", action="store_true", help="Show the portfolio and exit")
    p.add_argument("--reset", action="store_true", help="Delete the paper portfolio")
    p.add_argument("--deposit", type=float, default=0.0, help="Add cash before running")
    p.add_argument("--cost-preset", default="neobroker_1eur",
                   choices=sorted(cost_mod.PRESETS))
    p.add_argument("--max-positions", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.15)
    p.add_argument("--cost-multiple", type=float, default=2.0)
    args = p.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    model = cost_mod.get_preset(args.cost_preset)

    if args.reset:
        for f in (config.PAPER_DIR / "portfolio.json",
                  config.PAPER_DIR / "fills.parquet",
                  config.PAPER_DIR / "equity.parquet"):
            f.unlink(missing_ok=True)
        print("Paper portfolio reset.")
        return

    broker = PaperBroker.load_or_create(cost_model=model)
    broker.cost_model = model
    if args.deposit:
        broker.deposit(args.deposit)

    if args.report:
        marks = {}
        for s in list(broker.positions):
            try:
                marks[s] = float(intraday.read_intraday(s, args.interval)["close"].iloc[-1])
            except (FileNotFoundError, IndexError):
                pass
        report(broker, marks)
        return

    if args.refresh:
        intraday.download_and_cache_intraday(symbols, args.interval, args.days_back)

    cfg = EventConfig(max_positions=args.max_positions, min_score=args.min_score,
                      cost_multiple=args.cost_multiple)

    print(cost_mod.format_viability(model, n_positions=cfg.max_positions,
                                    holding_days=cfg.max_holding_bars / 6.5))
    print()

    bars = build_feature_bars(symbols, args.interval, cfg.max_holding_bars)
    if bars.empty:
        print(f"\nNo cached {args.interval} bars. Run with --refresh first.")
        return

    print(intraday.coverage(symbols, args.interval).to_string(index=False))

    if args.replay:
        replay(broker, bars, cfg, args.replay)
    else:
        latest = bars[bars["timestamp"] == bars["timestamp"].max()]
        marks = dict(zip(latest["symbol"], latest["close"]))
        equity = broker.equity(marks)
        notional = equity * cfg.position_fraction
        signals = find_events(latest, cfg, model, max(notional, 1.0),
                              holding_days=cfg.max_holding_bars / 6.5)
        print(f"\nBar {latest['timestamp'].iloc[0]}: {len(signals)} signal(s)")
        for s in signals:
            print(f"  {s.symbol} {'LONG' if s.direction > 0 else 'SHORT'} — {s.reason}")
        report(broker, marks)

    broker.save()


if __name__ == "__main__":
    main()
