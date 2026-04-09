from __future__ import annotations

try:
    from datamodel import Order, OrderDepth, TradingState  # type: ignore[import]
except ModuleNotFoundError:
    class Order:
        def __init__(self, symbol: str, price: int, quantity: int) -> None:
            self.symbol = symbol
            self.price = price
            self.quantity = quantity

    class OrderDepth:
        def __init__(self) -> None:
            self.buy_orders: dict[int, int] = {}
            self.sell_orders: dict[int, int] = {}

    class TradingState:
        def __init__(self) -> None:
            self.traderData: str = ""
            self.position: dict[str, int] = {}
            self.order_depths: dict[str, OrderDepth] = {}

from typing import Dict, List, Tuple
import json

LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

EMERALDS_FV = 10_000
EMERALDS_EDGE = 7

TOMATOES_ALPHA = 0.50
TOMATOES_EDGE = 6

SKEW_FACTOR = 0.5


class Trader:

    @staticmethod
    def _weighted_mid(od: OrderDepth) -> float:
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]
        return (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)

    @staticmethod
    def _capacities(product: str, state: TradingState) -> Tuple[int, int]:
        pos = state.position.get(product, 0)
        limit = LIMITS[product]
        return limit - pos, limit + pos

    @staticmethod
    def _sweep(
            product: str,
            fv: float,
            min_edge: float,
            od: OrderDepth,
            buy_cap: int,
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
            product: str,
            fv: float,
            edge: int,
            pos: int,
            buy_cap: int,
            sell_cap: int,
    ) -> List[Order]:
        limit = LIMITS[product]
        skew = -int(SKEW_FACTOR * edge * pos / limit)

        bid_price = round(fv) - edge + skew
        ask_price = round(fv) + edge + skew

        if bid_price >= ask_price:
            bid_price = round(fv) - 1
            ask_price = round(fv) + 1

        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid_price, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask_price, -sell_cap))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        ema: Dict[str, float] = {}
        if state.traderData:
            try:
                ema = json.loads(state.traderData)
            except (json.JSONDecodeError, ValueError, TypeError):
                ema = {}

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                continue

            pos = state.position.get(product, 0)
            buy_cap, sell_cap = self._capacities(product, state)
            orders: List[Order] = []

            if product == "EMERALDS":
                orders += self._make(
                    product, float(EMERALDS_FV), EMERALDS_EDGE,
                    pos, buy_cap, sell_cap,
                )

            elif product == "TOMATOES":
                weighted_mid = self._weighted_mid(od)
                ema[product] = (
                        TOMATOES_ALPHA * weighted_mid
                        + (1.0 - TOMATOES_ALPHA) * ema.get(product, weighted_mid)
                )
                fv = ema[product]

                take_threshold = TOMATOES_EDGE * 1.5
                orders, buy_cap, sell_cap = self._sweep(
                    product, fv, take_threshold, od, buy_cap, sell_cap,
                )

                orders += self._make(
                    product, fv, TOMATOES_EDGE,
                    pos, buy_cap, sell_cap,
                )

            if orders:
                result[product] = orders

        trader_data = json.dumps(ema)

        return result, 0, trader_data