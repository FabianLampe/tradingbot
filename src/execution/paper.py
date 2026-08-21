"""Paper broker — executes the strategy's orders against real prices, with
real costs, using no real money.

This is the honest way to find out whether the bot has an edge. A backtest
tells you what would have happened on data the model was tuned against; paper
trading tells you what happens on data that did not exist when you built it.
The difference between the two is usually most of the apparent edge.

What it models:
  - cash and positions, marked to market on request
  - every fill charged through `model.costs.CostModel`, so the fixed-fee
    penalty on small positions shows up exactly as it would live
  - fractional shares (optional) — at a 100 EUR/month account you cannot buy
    a whole share of most things, and pretending otherwise flatters the test
  - monthly deposits, so a savings-plan account can be simulated as it grows

What it deliberately does not model: partial fills, queue position, order
rejection, gaps through your stop, or a broker outage on the worst possible
day. Paper trading is optimistic by construction. Treat a paper result as an
upper bound on live performance, never as a forecast.

State lives in `data/paper/` and survives restarts, so the runner can be a
cron job rather than a long-running process.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import PAPER_DIR
from src.model.costs import CostModel, NEOBROKER_1EUR

log = logging.getLogger("trading_bot.execution.paper")

STATE_PATH = PAPER_DIR / "portfolio.json"
FILLS_PATH = PAPER_DIR / "fills.parquet"
EQUITY_PATH = PAPER_DIR / "equity.parquet"


class InsufficientFunds(RuntimeError):
    """Raised when an order would overdraw the account."""


@dataclass
class Position:
    symbol: str
    quantity: float          # negative = short
    avg_price: float
    opened_at: str

    @property
    def is_short(self) -> bool:
        return self.quantity < 0


@dataclass
class Fill:
    timestamp: str
    symbol: str
    side: str                # "buy" | "sell"
    quantity: float
    price: float
    notional: float
    cost: float              # spread + slippage + commission, currency units
    cash_delta: float
    realised_pnl: float = 0.0
    note: str = ""


@dataclass
class PaperBroker:
    """A cash account that fills market orders at the price you hand it."""

    cash: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    cost_model: CostModel = NEOBROKER_1EUR
    allow_fractional: bool = True
    allow_short: bool = False
    deposited: float = 0.0          # lifetime contributions, for honest return math
    realised_pnl: float = 0.0
    total_costs: float = 0.0

    # ---------------- account ----------------

    def deposit(self, amount: float, when: datetime | None = None) -> None:
        """Add cash. Tracked separately so returns are not inflated by deposits."""
        if amount <= 0:
            raise ValueError("deposit must be positive")
        self.cash += amount
        self.deposited += amount
        log.info("Deposit %.2f -> cash %.2f (lifetime %.2f)",
                 amount, self.cash, self.deposited)

    def position_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            px = prices.get(sym)
            if px is None:
                log.warning("[%s] no mark price — valuing at cost", sym)
                px = pos.avg_price
            total += pos.quantity * px
        return total

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus marked-to-market positions."""
        return self.cash + self.position_value(prices)

    def performance(self, prices: dict[str, float]) -> dict:
        """Return figures that separate performance from contributions.

        `profit` is equity minus everything ever paid in — the only number
        that answers "am I making money". A growing account balance on a
        savings plan is not a profit.
        """
        eq = self.equity(prices)
        profit = eq - self.deposited
        return {
            "equity": eq,
            "cash": self.cash,
            "positions_value": eq - self.cash,
            "deposited": self.deposited,
            "profit": profit,
            "return_on_deposits": profit / self.deposited if self.deposited else 0.0,
            "total_costs": self.total_costs,
            "costs_pct_of_deposits": (
                self.total_costs / self.deposited if self.deposited else 0.0
            ),
            "n_fills": len(self.fills),
            "n_positions": len(self.positions),
        }

    # ---------------- orders ----------------

    def _round_qty(self, qty: float) -> float:
        return qty if self.allow_fractional else float(int(qty))

    def target_quantity(self, price: float, notional: float) -> float:
        """Shares for a desired position size, respecting the fractional setting."""
        if price <= 0:
            return 0.0
        return self._round_qty(notional / price)

    def market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        timestamp: datetime | None = None,
        note: str = "",
    ) -> Fill | None:
        """Fill `quantity` shares at `price`, charging the cost model.

        Costs are paid in cash on top of the notional — that is what makes a
        1 EUR commission on a 10 EUR position visible instead of vanishing
        into a basis-point approximation.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        quantity = self._round_qty(abs(quantity))
        if quantity <= 0 or price <= 0:
            return None

        ts = (timestamp or datetime.now(tz=timezone.utc)).isoformat()
        notional = quantity * price
        cost = self.cost_model.cost_per_side(notional)
        signed = quantity if side == "buy" else -quantity
        existing = self.positions.get(symbol)

        if not self.allow_short:
            held = existing.quantity if existing else 0.0
            if held + signed < -1e-9:
                log.warning("[%s] short selling disabled — order skipped", symbol)
                return None

        cash_delta = (-notional if side == "buy" else notional) - cost
        if self.cash + cash_delta < -1e-9:
            raise InsufficientFunds(
                f"{side} {quantity:.4f} {symbol} @ {price:.2f} needs "
                f"{-cash_delta:.2f} but only {self.cash:.2f} is available"
            )

        realised = self._apply(symbol, signed, price, ts)
        self.cash += cash_delta
        self.total_costs += cost
        self.realised_pnl += realised

        fill = Fill(timestamp=ts, symbol=symbol, side=side, quantity=quantity,
                    price=price, notional=notional, cost=cost,
                    cash_delta=cash_delta, realised_pnl=realised, note=note)
        self.fills.append(fill)
        log.debug("%s %s %.4f @ %.2f (cost %.2f)", side, symbol, quantity, price, cost)
        return fill

    def _apply(self, symbol: str, signed_qty: float, price: float, ts: str) -> float:
        """Update the position book. Returns realised P&L for the closed part."""
        pos = self.positions.get(symbol)
        if pos is None:
            self.positions[symbol] = Position(symbol, signed_qty, price, ts)
            return 0.0

        realised = 0.0
        new_qty = pos.quantity + signed_qty
        closing = (pos.quantity > 0) != (signed_qty > 0)
        if closing:
            closed = min(abs(signed_qty), abs(pos.quantity))
            direction = 1.0 if pos.quantity > 0 else -1.0
            realised = closed * (price - pos.avg_price) * direction

        if abs(new_qty) < 1e-9:
            del self.positions[symbol]
        elif closing and (pos.quantity > 0) == (new_qty > 0):
            pos.quantity = new_qty          # partial close keeps the entry basis
        else:
            # adding to the position (or flipping through zero): re-average
            if closing:
                pos.avg_price = price
            else:
                total = pos.avg_price * pos.quantity + price * signed_qty
                pos.avg_price = total / new_qty
            pos.quantity = new_qty
        return realised

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
        note: str = "close",
    ) -> Fill | None:
        pos = self.positions.get(symbol)
        if pos is None:
            return None
        side = "sell" if pos.quantity > 0 else "buy"
        return self.market_order(symbol, side, abs(pos.quantity), price, timestamp, note)

    def close_all(self, prices: dict[str, float], timestamp: datetime | None = None) -> list[Fill]:
        out = []
        for sym in list(self.positions):
            px = prices.get(sym)
            if px is None:
                log.warning("[%s] no price to close against — position kept", sym)
                continue
            f = self.close_position(sym, px, timestamp)
            if f:
                out.append(f)
        return out

    # ---------------- persistence ----------------

    def record_equity(self, prices: dict[str, float], timestamp: datetime | None = None) -> dict:
        """Append one equity snapshot to `data/paper/equity.parquet`."""
        ts = timestamp or datetime.now(tz=timezone.utc)
        snap = {"timestamp": pd.Timestamp(ts), **self.performance(prices)}
        row = pd.DataFrame([snap])
        if EQUITY_PATH.exists():
            row = pd.concat([pd.read_parquet(EQUITY_PATH), row], ignore_index=True)
            row = row.drop_duplicates(subset=["timestamp"], keep="last")
        row.sort_values("timestamp").to_parquet(EQUITY_PATH, index=False)
        return snap

    def save(self, path: Path = STATE_PATH) -> Path:
        state = {
            "cash": self.cash,
            "deposited": self.deposited,
            "realised_pnl": self.realised_pnl,
            "total_costs": self.total_costs,
            "allow_fractional": self.allow_fractional,
            "allow_short": self.allow_short,
            "cost_model": asdict(self.cost_model),
            "positions": [asdict(p) for p in self.positions.values()],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
        # Always rewrite the fill log, even when empty. Skipping the write for a
        # fresh portfolio would leave a previous run's fills on disk, and the
        # next load() would silently adopt trades that never belonged to it.
        rows = [asdict(f) for f in self.fills]
        frame = (pd.DataFrame(rows) if rows
                 else pd.DataFrame(columns=[f.name for f in fields(Fill)]))
        frame.to_parquet(FILLS_PATH, index=False)
        return path

    @classmethod
    def load(cls, path: Path = STATE_PATH) -> "PaperBroker":
        if not path.exists():
            raise FileNotFoundError(f"No paper portfolio at {path}")
        s = json.loads(path.read_text())
        broker = cls(
            cash=s["cash"],
            positions={p["symbol"]: Position(**p) for p in s.get("positions", [])},
            cost_model=CostModel(**s["cost_model"]),
            allow_fractional=s.get("allow_fractional", True),
            allow_short=s.get("allow_short", False),
            deposited=s.get("deposited", 0.0),
            realised_pnl=s.get("realised_pnl", 0.0),
            total_costs=s.get("total_costs", 0.0),
        )
        if FILLS_PATH.exists():
            broker.fills = [Fill(**r) for r in
                            pd.read_parquet(FILLS_PATH).to_dict("records")]
        return broker

    @classmethod
    def load_or_create(cls, path: Path = STATE_PATH, **kw) -> "PaperBroker":
        try:
            return cls.load(path)
        except FileNotFoundError:
            log.info("Starting a fresh paper portfolio at %s", path)
            return cls(**kw)
