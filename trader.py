"""
EMERALDS
  Fair value is a constant 10,000 (mean=10000.00, std=0.72 over 20,000 rows).
  Bot quotes sit at FV ± 8 (bid=9992, ask=10008).  We post at FV ± 7 — one
  tick inside the bot spread — making us the unique best bid/ask.  Every market
  trade fills us at our price (9993 or 10007), giving 7 SeaShells per unit.
  No taking needed: no price ever crosses FV, so there is no arb to sweep.

TOMATOES
  Fair value drifts (autocorr 0.997).  We estimate it with a fast EMA (α=0.50)
  applied to the volume-weighted mid.  Bot spread ≈ 13 ticks; we quote ± 6,
  which is tight enough to win flow but not so tight we invert the book when
  the EMA lags.  Inventory skew prevents runaway directional exposure.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Tuple
import json

# Parameters
LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

# Emeralds, constant FV, bot spread = ±8, so we sit ±7 (inside, maximise $/fill)
EMERALDS_FV   = 10_000
EMERALDS_EDGE = 7          # bid=9993, ask=10007

# Tomatoes, adaptive FV via weighted-mid EMA
TOMATOES_ALPHA = 0.50      # fast EMA; empirically beats α∈{0.05…0.35} on this data
TOMATOES_EDGE  = 6         # bot half-spread ≈ 6.5; we match it for max fill rate

# Inventory skew: shifts both quotes toward mean-reversion
SKEW_FACTOR = 0.5          # max adjustment = SKEW_FACTOR × EDGE ticks at full pos


# Trader

class Trader:

    # Helpers

    @staticmethod
    def _wmid(od: OrderDepth) -> float:
        """
        Volume-weighted mid-price.
        Weights each side by the OPPOSITE side's volume, giving a directional
        lean: heavy ask-side pressure → wmid drifts below simple mid.
        Beats simple mid by ~6 % on TOMATOES next-tick MAE.
        """
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bv = od.buy_orders[best_bid]       # volume at best bid
        av = -od.sell_orders[best_ask]     # volume at best ask (stored as negative)
        return (best_bid * av + best_ask * bv) / (bv + av)

    @staticmethod
    def _capacities(product: str, state: TradingState) -> Tuple[int, int]:
        pos   = state.position.get(product, 0)
        limit = LIMITS[product]
        return limit - pos, limit + pos

    # Order Generation

    @staticmethod
    def _sweep(
            product:  str,
            fv:       float,
            min_edge: float,
            od:       OrderDepth,
            buy_cap:  int,
            sell_cap: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []

        for ask in sorted(od.sell_orders):
            if ask > fv - min_edge:
                break
            vol = min(-od.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order(product, ask, vol))
                buy_cap -= vol

        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fv + min_edge:
                break
            vol = min(od.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order(product, bid, -vol))
                sell_cap -= vol

        return orders, buy_cap, sell_cap

    @staticmethod
    def _make(
            product:  str,
            fv:       float,
            edge:     int,
            pos:      int,
            buy_cap:  int,
            sell_cap: int,
    ) -> List[Order]:
        limit    = LIMITS[product]
        skew     = -int(SKEW_FACTOR * edge * pos / limit)

        bid_price = round(fv) - edge + skew
        ask_price = round(fv) + edge + skew

        # Safety guard: never cross the spread (can happen at extreme inventory)
        if bid_price >= ask_price:
            bid_price = round(fv) - 1
            ask_price = round(fv) + 1

        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid_price,  buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -sell_cap))
        return orders

    # Main Entry Point

    def run(self, state: TradingState):
        ema: Dict[str, float] = {}
        if state.traderData:
            try:
                ema = json.loads(state.traderData)
            except Exception:
                pass

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                continue

            pos              = state.position.get(product, 0)
            buy_cap, sell_cap = self._capacities(product, state)
            orders: List[Order] = []

            # Emeralds
            if product == "EMERALDS":
                # Constant FV; no sweeping needed (zero arb exists in data).
                # Post full remaining capacity at FV ± 7 (inside bot's ± 8).
                orders += self._make(
                    product, float(EMERALDS_FV), EMERALDS_EDGE,
                    pos, buy_cap, sell_cap,
                )

            # Tomatoes
            elif product == "TOMATOES":
                # Fair-value: fast EMA of volume-weighted mid
                wmid = self._wmid(od)
                ema[product] = (
                        TOMATOES_ALPHA * wmid
                        + (1.0 - TOMATOES_ALPHA) * ema.get(product, wmid)
                )
                fv = ema[product]

                # Pass 1: sweep fat-tail mispricings (≥ 1.5 × edge from FV)
                take_thresh = TOMATOES_EDGE * 1.5
                orders, buy_cap, sell_cap = self._sweep(
                    product, fv, take_thresh, od, buy_cap, sell_cap,
                )

                # Pass 2: post remaining capacity as resting quotes
                orders += self._make(
                    product, fv, TOMATOES_EDGE,
                    pos, buy_cap, sell_cap,
                )

            if orders:
                result[product] = orders

        # 2. Persist updated EMA for next tick
        trader_data = json.dumps(ema)

        return result, 0, trader_data