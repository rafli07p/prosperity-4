from datamodel import OrderDepth, TradingState, Order
from typing import Any
import json
import math

# Position limits per the exchange specification.
POS_LIMIT: int = 80

# Osmium reverts to this level across all 3 observed days.
# Empirical mean of microprice: 10,000.19 (N = 27,696 observations).
OSMIUM_FAIR_ANCHOR: float = 10_000.0

# EMA decay for adaptive fair value.  α = 2/(N+1).
# N = 200 gives α ≈ 0.01 — slow enough to ignore microstructure noise
# (tick σ = 1.93) while tracking any intraday regime shift.
OSMIUM_EMA_SPAN: int = 200
OSMIUM_EMA_ALPHA: float = 2.0 / (OSMIUM_EMA_SPAN + 1)

# Market-making half-spread.  The observed full spread is 16 ticks.
# Posting at fair ± 4 places us well inside the market (at the ~50th
# percentile of the bid-ask range), ensuring queue priority while
# retaining 8 ticks of edge per round-trip before inventory costs.
OSMIUM_MM_HALF_SPREAD: int = 4

# Inventory skew coefficient.  For every 10 units of net position,
# shift both bid and ask by 1 tick against the position to accelerate
# mean reversion of inventory.  Bounded at ±3 ticks to prevent
# crossing our own quotes.
OSMIUM_SKEW_COEFF: float = 0.1
OSMIUM_MAX_SKEW: int = 3


class Trader:
    """Stateless per tick; all persistence goes through traderData JSON."""

    def run(
            self, state: TradingState
    ) -> tuple[dict[str, list[Order]], int, str]:
        result: dict[str, list[Order]] = {}
        conversions: int = 0
        store = _load_store(state.traderData)

        result["ASH_COATED_OSMIUM"] = _trade_osmium(state, store)
        result["INTARIAN_PEPPER_ROOT"] = _trade_pepper(state, store)

        return result, conversions, json.dumps(store)


def _trade_pepper(state: TradingState, store: dict) -> list[Order]:
    sym = "INTARIAN_PEPPER_ROOT"
    depth = state.order_depths.get(sym)
    if depth is None:
        return []

    pos: int = state.position.get(sym, 0)
    capacity: int = POS_LIMIT - pos
    if capacity <= 0:
        return []

    orders: list[Order] = []

    # Sweep all visible ask levels — price is irrelevant because
    # expected drift dominates any spread cost.
    if depth.sell_orders:
        for price in sorted(depth.sell_orders):
            if capacity <= 0:
                break
            qty = min(-depth.sell_orders[price], capacity)
            orders.append(Order(sym, price, qty))
            capacity -= qty

    # Post a passive bid for residual capacity.
    if capacity > 0:
        ref = _best_ask(depth) or _best_bid(depth)
        if ref is not None:
            bid = (ref - 1) if depth.sell_orders else (ref + 2)
            orders.append(Order(sym, bid, capacity))

    return orders


def _trade_osmium(state: TradingState, store: dict) -> list[Order]:
    sym = "ASH_COATED_OSMIUM"
    depth = state.order_depths.get(sym)
    if depth is None:
        return []

    pos: int = state.position.get(sym, 0)

    # Fair value
    fair = _update_osmium_fair(depth, store)
    fair_r = round(fair)

    buy_cap = POS_LIMIT - pos
    sell_cap = POS_LIMIT + pos
    orders: list[Order] = []

    # Aggress on mispriced levels
    buy_cap, sell_cap = _aggress_osmium(
        sym, depth, fair_r, pos, buy_cap, sell_cap, orders
    )

    # Post market-making quotes
    _post_osmium_quotes(sym, fair_r, pos, buy_cap, sell_cap, orders)

    return orders


def _update_osmium_fair(depth: OrderDepth, store: dict) -> float:
    bb, ba = _best_bid(depth), _best_ask(depth)
    prev = store.get("osm_fair", OSMIUM_FAIR_ANCHOR)

    if bb is None or ba is None:
        return prev

    bv = depth.buy_orders.get(bb, 0)
    av = -depth.sell_orders.get(ba, 0)      # sell volumes are negative
    denom = bv + av

    # Microprice: weight each side by the opposite sidesvolume.
    micro = (bb * av + ba * bv) / denom if denom > 0 else (bb + ba) / 2

    fair = OSMIUM_EMA_ALPHA * micro + (1 - OSMIUM_EMA_ALPHA) * prev
    store["osm_fair"] = fair
    return fair


def _aggress_osmium(
        sym: str,
        depth: OrderDepth,
        fair: int,
        pos: int,
        buy_cap: int,
        sell_cap: int,
        orders: list[Order],
) -> tuple[int, int]:
    """
    Sweep mispriced resting orders.

    Buy  every ask where price < fair.
    Sell every bid where price > fair.
    At price == fair, only aggress to reduce |inventory| (no edge,
    but reduces gamma risk).
    """
    # Buy side: sweep cheap asks
    if depth.sell_orders:
        for price in sorted(depth.sell_orders):
            if buy_cap <= 0:
                break
            if price < fair:
                qty = min(-depth.sell_orders[price], buy_cap)
                orders.append(Order(sym, price, qty))
                buy_cap -= qty
            elif price == fair and pos < 0:
                # Flatten short inventory at fair — no alpha, pure risk reduction.
                qty = min(-depth.sell_orders[price], buy_cap, abs(pos))
                if qty > 0:
                    orders.append(Order(sym, price, qty))
                    buy_cap -= qty

    # Sell side: hit expensive bids
    if depth.buy_orders:
        for price in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0:
                break
            if price > fair:
                qty = min(depth.buy_orders[price], sell_cap)
                orders.append(Order(sym, price, -qty))
                sell_cap -= qty
            elif price == fair and pos > 0:
                qty = min(depth.buy_orders[price], sell_cap, pos)
                if qty > 0:
                    orders.append(Order(sym, price, -qty))
                    sell_cap -= qty

    return buy_cap, sell_cap


def _post_osmium_quotes(
        sym: str,
        fair: int,
        pos: int,
        buy_cap: int,
        sell_cap: int,
        orders: list[Order],
) -> None:
    """
    Post inventory-skewed bid/ask around fair value.

    Reservation price:  r = fair − skew(pos)
    Bid:                r − δ
    Ask:                r + δ

    The skew pushes both quotes against our inventory direction,
    making it cheaper for counterparties to trade us back toward flat.
    """
    skew = -_clamp(
        round(pos * OSMIUM_SKEW_COEFF),
        -OSMIUM_MAX_SKEW,
        OSMIUM_MAX_SKEW,
    )

    bid_px = fair - OSMIUM_MM_HALF_SPREAD + skew
    ask_px = fair + OSMIUM_MM_HALF_SPREAD + skew

    # Sanity: never let our quotes cross.
    if bid_px >= ask_px:
        bid_px = fair - 1
        ask_px = fair + 1

    if buy_cap > 0:
        orders.append(Order(sym, bid_px, buy_cap))
    if sell_cap > 0:
        orders.append(Order(sym, ask_px, -sell_cap))


#  Utility functions

def _best_bid(depth: OrderDepth) -> int | None:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> int | None:
    return min(depth.sell_orders) if depth.sell_orders else None

def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def _load_store(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}