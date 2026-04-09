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

LIMITS: Dict[str, int] = {"EMERALDS": 80, "TOMATOES": 80}

EM_FV = 10_000
EM_EDGE = 7
EM_TAKE = 1
EM_SKEW = 0.15

TOM_ALPHA = 0.50
TOM_TAKE = 1
TOM_MIN_EDGE = 4
TOM_SPREAD_OFF = 2
TOM_SKEW = 0.40

INNER_RATIO = 0.65


class Trader:

    @staticmethod
    def _weighted_mid(od: OrderDepth) -> float:
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = od.buy_orders[best_bid]
        ask_vol = -od.sell_orders[best_ask]
        total = bid_vol + ask_vol
        if total == 0:
            return (best_bid + best_ask) / 2.0
        return (best_bid * ask_vol + best_ask * bid_vol) / total

    @staticmethod
    def _caps(product: str, state: TradingState) -> Tuple[int, int]:
        pos = state.position.get(product, 0)
        lim = LIMITS[product]
        return lim - pos, lim + pos

    @staticmethod
    def _sweep(
        product: str,
        fv: float,
        threshold: float,
        od: OrderDepth,
        buy_cap: int,
        sell_cap: int,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        for ask in sorted(od.sell_orders):
            if ask > fv - threshold or buy_cap <= 0:
                break
            vol = min(-od.sell_orders[ask], buy_cap)
            if vol > 0:
                orders.append(Order(product, ask, vol))
                buy_cap -= vol
        for bid in sorted(od.buy_orders, reverse=True):
            if bid < fv + threshold or sell_cap <= 0:
                break
            vol = min(od.buy_orders[bid], sell_cap)
            if vol > 0:
                orders.append(Order(product, bid, -vol))
                sell_cap -= vol
        return orders, buy_cap, sell_cap

    @staticmethod
    def _make_single(
        product: str,
        fv: float,
        edge: int,
        skew_factor: float,
        pos: int,
        buy_cap: int,
        sell_cap: int,
    ) -> List[Order]:
        lim = LIMITS[product]
        skew = -int(skew_factor * edge * pos / lim)
        bid = round(fv) - edge + skew
        ask = round(fv) + edge + skew

        if bid >= ask:
            bid = round(fv) - 1
            ask = round(fv) + 1

        orders: List[Order] = []
        if buy_cap > 0:
            orders.append(Order(product, bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(product, ask, -sell_cap))
        return orders

    @staticmethod
    def _make_two_level(
        product: str,
        fv: float,
        edge_inner: int,
        edge_outer: int,
        skew_factor: float,
        pos: int,
        buy_cap: int,
        sell_cap: int,
    ) -> List[Order]:
        lim = LIMITS[product]
        skew = -int(skew_factor * edge_inner * pos / lim)

        bid1 = round(fv) - edge_inner + skew
        ask1 = round(fv) + edge_inner + skew
        bid2 = round(fv) - edge_outer + skew
        ask2 = round(fv) + edge_outer + skew

        if bid1 >= ask1:
            orders: List[Order] = []
            if buy_cap > 0:
                orders.append(Order(product, round(fv) - 1, buy_cap))
            if sell_cap > 0:
                orders.append(Order(product, round(fv) + 1, -sell_cap))
            return orders

        orders: List[Order] = []
        if buy_cap > 0:
            q1 = max(1, int(buy_cap * INNER_RATIO)) if buy_cap >= 2 else buy_cap
            q2 = buy_cap - q1
            orders.append(Order(product, bid1, q1))
            if q2 > 0:
                orders.append(Order(product, bid2, q2))
        if sell_cap > 0:
            q1 = max(1, int(sell_cap * INNER_RATIO)) if sell_cap >= 2 else sell_cap
            q2 = sell_cap - q1
            orders.append(Order(product, ask1, -q1))
            if q2 > 0:
                orders.append(Order(product, ask2, -q2))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        saved: Dict[str, float] = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except (json.JSONDecodeError, ValueError, TypeError):
                saved = {}

        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            if not od.buy_orders or not od.sell_orders:
                continue

            pos = state.position.get(product, 0)
            buy_cap, sell_cap = self._caps(product, state)
            orders: List[Order] = []

            if product == "EMERALDS":
                sweep, buy_cap, sell_cap = self._sweep(
                    product, float(EM_FV), EM_TAKE, od, buy_cap, sell_cap,
                )
                orders += sweep
                orders += self._make_single(
                    product, float(EM_FV), EM_EDGE, EM_SKEW,
                    pos, buy_cap, sell_cap,
                )

            elif product == "TOMATOES":
                wmid = self._weighted_mid(od)
                saved[product] = (
                    TOM_ALPHA * wmid
                    + (1.0 - TOM_ALPHA) * saved.get(product, wmid)
                )
                fv = saved[product]

                sweep, buy_cap, sell_cap = self._sweep(
                    product, fv, TOM_TAKE, od, buy_cap, sell_cap,
                )
                orders += sweep

                best_bid = max(od.buy_orders)
                best_ask = min(od.sell_orders)
                half_spread = (best_ask - best_bid) // 2
                make_edge = max(TOM_MIN_EDGE, half_spread - TOM_SPREAD_OFF)

                orders += self._make_two_level(
                    product, fv, make_edge, make_edge + 2,
                    TOM_SKEW, pos, buy_cap, sell_cap,
                )

            if orders:
                result[product] = orders

        return result, 0, json.dumps(saved)
