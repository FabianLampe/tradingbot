"""Realistic trading costs.

The old backtest used one flat number (`cost_bps`) applied as a fraction of
the position. That hides the single most important fact about a small
account: **a fixed fee per order is a percentage cost that explodes as the
position shrinks.** One euro of commission on a 1000 EUR position is 0.1%.
The same euro on a 10 EUR position is 10%, and no signal survives that.

So costs are modelled in currency, not in basis points, and converted to a
return only once the position's notional is known:

    total = spread + slippage + commission (+ borrow for shorts)

  - **spread**    — half the quoted bid/ask, paid on entry and on exit. You
                    cross it every time you trade at market.
  - **slippage**  — the price moving away from you between decision and
                    fill. Grows with order size relative to volume; here it
                    is a flat per-side estimate you set per liquidity tier.
  - **commission**— broker fee: fixed per order, percentage of notional, or
                    a percentage with a minimum. All three exist in the wild.
  - **borrow**    — shorts pay a daily borrow fee. Long-only strategies can
                    leave it at zero.

Everything is per *side*. A round trip pays entry and exit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    """Cost assumptions for one instrument class at one broker."""

    spread_bps: float = 5.0            # half-spread crossed per side
    slippage_bps: float = 2.0          # adverse move between decision and fill
    fee_per_order: float = 0.0         # fixed commission, currency units
    fee_pct: float = 0.0               # commission as a fraction of notional
    min_fee: float = 0.0               # floor on the commission per order
    borrow_bps_per_day: float = 0.0    # short borrow, charged per day held
    name: str = "custom"

    # ---------- per-trade ----------

    def commission(self, notional: float) -> float:
        """Broker fee for one order, in currency units."""
        fee = self.fee_per_order + self.fee_pct * abs(notional)
        return max(fee, self.min_fee) if self.min_fee else fee

    def cost_per_side(self, notional: float) -> float:
        """Spread + slippage + commission for a single entry or exit."""
        notional = abs(notional)
        market = notional * (self.spread_bps + self.slippage_bps) / 10_000
        return market + self.commission(notional)

    def borrow_cost(self, notional: float, holding_days: float) -> float:
        if self.borrow_bps_per_day <= 0:
            return 0.0
        return abs(notional) * self.borrow_bps_per_day / 10_000 * max(holding_days, 0)

    def round_trip(
        self,
        notional: float,
        holding_days: float = 0.0,
        side: str = "long",
    ) -> float:
        """Full cost of opening and closing one position, in currency units."""
        cost = 2 * self.cost_per_side(notional)
        if side == "short":
            cost += self.borrow_cost(notional, holding_days)
        return cost

    def round_trip_fraction(
        self,
        notional: float,
        holding_days: float = 0.0,
        side: str = "long",
    ) -> float:
        """Round-trip cost as a fraction of the position.

        This is the number that decides whether a strategy is viable at a
        given account size. Compare it against the *average* move you expect
        to capture: if a trade is worth 0.4% and the round trip costs 0.6%,
        no amount of model accuracy saves it.
        """
        notional = abs(notional)
        if notional <= 0:
            return 0.0
        return self.round_trip(notional, holding_days, side) / notional

    # ---------- aggregate ----------

    def annual_drag(
        self,
        equity: float,
        holding_days: float,
        n_positions: int,
        side: str = "long",
    ) -> float:
        """Yearly return drag for a strategy that stays fully invested.

        Assumes the whole book turns over every `holding_days`, which is what
        a fixed-horizon Top-N strategy does.
        """
        if equity <= 0 or holding_days <= 0 or n_positions <= 0:
            return 0.0
        notional = equity / n_positions
        turnovers = TRADING_DAYS_PER_YEAR / holding_days
        per_turnover = n_positions * self.round_trip(notional, holding_days, side)
        return turnovers * per_turnover / equity

    def breakeven_move_bps(self, notional: float, holding_days: float = 0.0) -> float:
        """How far the price must move, in bps, just to cover the round trip."""
        return self.round_trip_fraction(notional, holding_days) * 10_000

    def with_(self, **changes) -> "CostModel":
        """Copy with fields overridden (frozen dataclass convenience)."""
        return replace(self, **changes)


# ---------------------------------------------------------------------------
# Presets. Verify against your own broker's fee schedule before trusting them —
# these are plausible orders of magnitude, not quotes.
# ---------------------------------------------------------------------------

#: Liquid large-cap ETFs (SPY, QQQ) at a zero-commission broker. Best case.
ETF_ZERO_COMMISSION = CostModel(
    spread_bps=2.0, slippage_bps=1.0, fee_per_order=0.0, name="etf_zero_commission",
)

#: German neobroker: ~1 EUR per order, decent spreads during main session.
NEOBROKER_1EUR = CostModel(
    spread_bps=5.0, slippage_bps=3.0, fee_per_order=1.0, name="neobroker_1eur",
)

#: Interactive-Brokers-style: percentage fee with a per-order minimum.
IBKR_STYLE = CostModel(
    spread_bps=3.0, slippage_bps=2.0, fee_pct=0.0005, min_fee=1.0, name="ibkr_style",
)

#: Single stocks outside the mega-caps, or trading outside the main session.
ILLIQUID = CostModel(
    spread_bps=15.0, slippage_bps=10.0, fee_per_order=1.0,
    borrow_bps_per_day=1.0, name="illiquid",
)

#: What the original backtest assumed: 5 bps per side, nothing else.
LEGACY_FLAT = CostModel(
    spread_bps=5.0, slippage_bps=0.0, fee_per_order=0.0, name="legacy_flat",
)

PRESETS: dict[str, CostModel] = {
    m.name: m for m in (
        ETF_ZERO_COMMISSION, NEOBROKER_1EUR, IBKR_STYLE, ILLIQUID, LEGACY_FLAT,
    )
}


def get_preset(name: str) -> CostModel:
    if name not in PRESETS:
        raise ValueError(f"Unknown cost preset {name!r}. Available: {sorted(PRESETS)}")
    return PRESETS[name]


def viability_table(
    model: CostModel,
    equities: tuple[float, ...] = (600, 1_200, 5_000, 25_000, 100_000),
    n_positions: int = 10,
    holding_days: float = 5.0,
) -> "list[dict]":
    """Round-trip cost and annual drag across account sizes.

    Printed by the backtest and the paper runner so the size problem is
    visible on every run rather than buried in a docstring.
    """
    rows = []
    for eq in equities:
        notional = eq / n_positions
        rows.append({
            "equity": eq,
            "position_size": notional,
            "round_trip_pct": model.round_trip_fraction(notional, holding_days),
            "breakeven_bps": model.breakeven_move_bps(notional, holding_days),
            "annual_drag": model.annual_drag(eq, holding_days, n_positions),
        })
    return rows


def format_viability(
    model: CostModel,
    n_positions: int = 10,
    holding_days: float = 5.0,
    **kw,
) -> str:
    """Human-readable version of `viability_table`."""
    lines = [
        f"Cost model '{model.name}': {model.spread_bps:g} bps spread + "
        f"{model.slippage_bps:g} bps slippage per side, "
        f"fee {model.fee_per_order:g} + {model.fee_pct:.4%} (min {model.min_fee:g})",
        f"{n_positions} positions, {holding_days:g}-day holding period",
        f"{'Equity':>10} {'Position':>10} {'Round trip':>11} {'Breakeven':>11} {'Drag p.a.':>10}",
    ]
    for r in viability_table(model, n_positions=n_positions, holding_days=holding_days, **kw):
        lines.append(
            f"{r['equity']:>10,.0f} {r['position_size']:>10,.1f} "
            f"{r['round_trip_pct']:>10.2%} {r['breakeven_bps']:>10.0f}b "
            f"{r['annual_drag']:>9.1%}"
        )
    return "\n".join(lines)
