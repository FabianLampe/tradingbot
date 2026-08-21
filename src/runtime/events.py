"""Event-driven strategy: trade on a trigger, not on a clock.

The fixed-horizon Top-N strategy rebalances every N days whether or not
anything happened. That guarantees ~50 round trips a year, and — per
`model.costs` — a drag of several percent before any signal is considered.
Most of those trades are the model ranking noise slightly above other noise.

This module trades only when something identifiable occurs:

    trigger = a news event on the symbol
              AND the model agreeing with its direction
              AND the expected move clearing the round-trip cost

The third condition is the one that changes the economics. A trade worth an
expected 0.3% is not worth taking when the round trip costs 0.5%, no matter
how confident the model is. `model.costs` knows the cost at the account's
actual position size, so the filter tightens automatically as the account
shrinks — a small account simply takes fewer, larger-edge trades.

Exits are explicit: a maximum holding period, a profit target, and a stop.
Nothing is held indefinitely, and nothing is closed just because the calendar
rolled over.

The signal is deliberately conservative. Fewer trades with a real trigger beat
a thousand trades with none — that is the whole thesis of this module.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.model.costs import CostModel, NEOBROKER_1EUR

log = logging.getLogger("trading_bot.runtime.events")


@dataclass
class EventConfig:
    """Thresholds for entering and leaving a trade."""

    # --- entry ---
    min_score: float = 0.15
    """Model conviction, |P(up) - P(down)|. Below this the model is guessing."""

    min_articles: int = 1
    """How many articles must exist for the bar to count as an event at all."""

    min_sentiment: float = 0.25
    """|sentiment| needed to call it news rather than routine coverage."""

    require_agreement: bool = True
    """Only trade when model score and news sentiment point the same way.
    Disagreement means one of the two is wrong and we do not know which."""

    cost_multiple: float = 2.0
    """Expected move must exceed this multiple of the round-trip cost.
    At 1.0 you break even on average and keep none of it; 2.0 leaves a margin
    for the estimate being optimistic, which it usually is."""

    # --- exit ---
    max_holding_bars: int = 6
    """Hard time stop. On hourly bars, 6 is roughly one session."""

    take_profit: float = 0.02
    stop_loss: float = 0.01
    """Asymmetric on purpose: cut losers faster than winners. The stop is what
    keeps a single gap from erasing a month of small gains."""

    # --- sizing ---
    max_positions: int = 3
    """Small accounts cannot diversify meaningfully; concentrating in the few
    highest-conviction events beats spreading fixed fees across ten names."""

    position_fraction: float = 0.25
    """Fraction of equity per position. max_positions * this should be <= 1."""

    allow_short: bool = False


@dataclass
class Signal:
    timestamp: pd.Timestamp
    symbol: str
    direction: int              # +1 long, -1 short
    score: float
    sentiment: float
    n_articles: int
    expected_move: float
    cost_hurdle: float
    reason: str


def expected_move_from_score(score: float, typical_move: float) -> float:
    """Translate a classifier score into an expected return.

    `score` is P(up) - P(down) in [-1, 1]; `typical_move` is the average
    absolute move over the horizon (estimate it from realised volatility).
    The product is a crude expectation, and crude is the right register here:
    a classifier trained on ±1% buckets does not carry magnitude information
    (see `model.predictor.predict_score`), so anything more precise would be
    false precision.
    """
    return float(score) * float(typical_move)


def find_events(
    bars: pd.DataFrame,
    config: EventConfig,
    cost_model: CostModel = NEOBROKER_1EUR,
    position_notional: float = 1_000.0,
    holding_days: float = 0.25,
) -> list[Signal]:
    """Scan a feature frame for tradeable events.

    `bars` needs columns: timestamp, symbol, score, sent_mean, n_articles,
    typical_move. One row per (symbol, bar).
    """
    required = {"timestamp", "symbol", "score", "sent_mean", "n_articles", "typical_move"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"find_events needs columns {sorted(missing)}")
    if bars.empty:
        return []

    hurdle = cost_model.round_trip_fraction(position_notional, holding_days) * config.cost_multiple

    df = bars.copy()
    df["direction"] = np.sign(df["score"]).astype(int)
    df["expected_move"] = [
        expected_move_from_score(s, m) for s, m in zip(df["score"], df["typical_move"])
    ]

    is_event = (df["n_articles"] >= config.min_articles) & \
               (df["sent_mean"].abs() >= config.min_sentiment)
    convinced = df["score"].abs() >= config.min_score
    worth_it = df["expected_move"].abs() >= hurdle
    keep = is_event & convinced & worth_it

    if config.require_agreement:
        keep &= np.sign(df["score"]) == np.sign(df["sent_mean"])
    if not config.allow_short:
        keep &= df["direction"] > 0

    hits = df[keep]
    log.info(
        "Events: %d of %d bars pass (news %d, conviction %d, cost hurdle %.2f%% %d)",
        len(hits), len(df), int(is_event.sum()), int(convinced.sum()),
        hurdle * 100, int(worth_it.sum()),
    )

    return [
        Signal(
            timestamp=pd.Timestamp(r["timestamp"]),
            symbol=str(r["symbol"]),
            direction=int(r["direction"]),
            score=float(r["score"]),
            sentiment=float(r["sent_mean"]),
            n_articles=int(r["n_articles"]),
            expected_move=float(r["expected_move"]),
            cost_hurdle=hurdle,
            reason=(f"score {r['score']:+.2f}, sentiment {r['sent_mean']:+.2f}, "
                    f"{int(r['n_articles'])} articles, "
                    f"expected {r['expected_move']:+.2%} vs hurdle {hurdle:.2%}"),
        )
        for _, r in hits.iterrows()
    ]


def signal_funnel(
    bars: pd.DataFrame,
    config: EventConfig,
    cost_model: CostModel = NEOBROKER_1EUR,
    position_notional: float = 1_000.0,
    holding_days: float = 0.25,
) -> dict:
    """How many bars survive each filter, and what the binding constraint is.

    "No trades" has two very different causes: the signal never fired, or the
    signal fired and could not clear the cost hurdle. The first is a modelling
    problem, the second is an account-size problem, and they need opposite
    responses — so the runner reports which one it is instead of printing a
    bare zero.
    """
    if bars.empty:
        return {"total": 0}

    hurdle = cost_model.round_trip_fraction(position_notional, holding_days) * config.cost_multiple
    expected = (bars["score"] * bars["typical_move"]).abs()
    agree = np.sign(bars["score"]) == np.sign(bars["sent_mean"])

    out = {
        "total": len(bars),
        "has_news": int((bars["n_articles"] >= config.min_articles).sum()),
        "sentiment_strong": int((bars["sent_mean"].abs() >= config.min_sentiment).sum()),
        "model_convinced": int((bars["score"].abs() >= config.min_score).sum()),
        "direction_agrees": int(agree.sum()),
        "clears_cost": int((expected >= hurdle).sum()),
        "hurdle": hurdle,
        "position_notional": position_notional,
        "expected_median": float(expected.median()),
        "expected_max": float(expected.max()),
    }
    out["cost_is_binding"] = out["clears_cost"] == 0 and out["direction_agrees"] > 0
    out["max_viable_at_this_size"] = out["expected_max"] >= hurdle
    return out


def format_funnel(f: dict) -> str:
    """Readable funnel, with the diagnosis spelled out."""
    if not f.get("total"):
        return "No bars to evaluate."
    lines = [
        f"Signal funnel over {f['total']:,} bars:",
        f"  news present        {f['has_news']:>8,}",
        f"  sentiment strong    {f['sentiment_strong']:>8,}",
        f"  model convinced     {f['model_convinced']:>8,}",
        f"  direction agrees    {f['direction_agrees']:>8,}",
        f"  clears cost hurdle  {f['clears_cost']:>8,}   "
        f"(hurdle {f['hurdle']:.2%} at {f['position_notional']:,.0f} EUR/position)",
        f"  expected move: median {f['expected_median']:.3%}, best {f['expected_max']:.3%}",
    ]
    if f["cost_is_binding"]:
        lines.append("")
        if not f["max_viable_at_this_size"]:
            lines.append(
                "  -> COSTS ARE BINDING. Even the single best signal in this sample "
                "cannot\n     cover the round trip at this position size. No model "
                "improvement fixes\n     this — only bigger positions, a cheaper "
                "broker, or a longer horizon."
            )
        else:
            lines.append("  -> Costs reject most signals; only the strongest few clear.")
    elif f["direction_agrees"] == 0:
        lines.append("\n  -> The signal never fired. This is a modelling problem, "
                     "not a cost problem.")
    return "\n".join(lines)


def exit_reason(
    entry_price: float,
    current_price: float,
    direction: int,
    bars_held: int,
    config: EventConfig,
) -> str | None:
    """Why this position should close now, or None to keep holding."""
    if entry_price <= 0:
        return None
    move = (current_price - entry_price) / entry_price * direction
    if move >= config.take_profit:
        return "take_profit"
    if move <= -config.stop_loss:
        return "stop_loss"
    if bars_held >= config.max_holding_bars:
        return "time_stop"
    return None


def typical_move_from_bars(
    close: pd.Series,
    horizon_bars: int,
    lookback: int = 100,
) -> pd.Series:
    """Rolling estimate of the average absolute move over `horizon_bars`.

    Feeds the cost hurdle: it is the difference between "the model is bullish"
    and "the model is bullish and the instrument actually moves enough for
    that to pay for the spread".
    """
    log_ret = np.log(close / close.shift(1))
    per_bar_vol = log_ret.rolling(lookback, min_periods=max(10, lookback // 5)).std()
    return (per_bar_vol * np.sqrt(max(horizon_bars, 1))).fillna(0.0)
