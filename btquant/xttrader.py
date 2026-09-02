# -*- coding: utf-8 -*-
"""
xtquant.xttrader 影子实现。

把SilverQuant XtQuantTrader 的方法代理到大 QMT helper HTTP 网关：
  - connect / subscribe        -> /health + 启动回调轮询线程
  - order_stock / _async       -> /place_order
  - cancel_order_stock / _async -> /cancel_order
  - query_stock_asset           -> /account
  - query_stock_orders          -> /orders
  - query_stock_positions       -> /positions
  - query_stock_order           -> /orders?order_id=
  - query_ipo_data / query_new_purchase_limit -> 返回空（helper 未提供）

回调推送：由于 helper 是 HTTP 拉取模型，order_stock_async 后立即触发
on_order_stock_async_response；subscribe 后启动后台线程轮询 /orders 与 /trades，
检测新增/变化的订单和成交后调用 callback.on_stock_order / on_stock_trade。
"""
import logging
import threading
import time
from typing import Any, Dict, List, Optional

from . import xtconstant
from ._bridge import GatewayClient, GatewayError, get_client, to_qmt_code, from_jq_code
from .xttype import (
    StockAccount,
    XtAsset,
    XtPosition,
    XtOrder,
    XtTrade,
    XtOrderResponse,
    XtCancelOrderResponse,
    XtQuantTraderCallback,
    build_asset,
    build_position,
    build_order,
    build_trade,
)

_LOGGER = logging.getLogger("xtquant.xttrader")

# miniqmt xtconstant -> 大 QMT passorder prType（两套枚举不同，见迅投知识库）
_MINIQMT_TO_PASSORDER_PR_TYPE = {
    xtconstant.FIX_PRICE: 11,
    xtconstant.LATEST_PRICE: 5,
    xtconstant.MARKET_PEER_PRICE_FIRST: 44,
    xtconstant.MARKET_PEER_PRICE_LAST: 45,
    xtconstant.MARKET_SH_BEST_5_CANCEL: 42,
    xtconstant.MARKET_SZ_CONVERT_5_CANCEL: 47,
    xtconstant.MARKET_SH_EDGE_5_CANCEL: 43,
}
# passorder 市价类 prType（price 可为 0，表示保护限价取涨跌停）
_PASSORDER_MARKET_PR_TYPES = {5, 42, 43, 44, 45, 46, 47, 48}

# miniqmt 侧市价 price_type（下单前会映射为 passorder prType）
_MARKET_PRICE_TYPES = {
    xtconstant.MARKET_SZ_CONVERT_5_CANCEL,
    xtconstant.MARKET_PEER_PRICE_FIRST,
    xtconstant.MARKET_PEER_PRICE_LAST,
    xtconstant.MARKET_SH_BEST_5_CANCEL,
    xtconstant.MARKET_SH_EDGE_5_CANCEL,
    xtconstant.LATEST_PRICE,
}


def _to_passorder_pr_type(price_type: int) -> int:
    """miniqmt 的 price_type 转为大 QMT passorder 的 prType。"""
    pt = int(price_type)
    return _MINIQMT_TO_PASSORDER_PR_TYPE.get(pt, pt)


def _is_market_price_type(price_type: int) -> bool:
    pt = int(price_type)
    if pt in _MARKET_PRICE_TYPES:
        return True
    return _to_passorder_pr_type(pt) in _PASSORDER_MARKET_PR_TYPES


def _side_to_order_type(side: Any) -> int:
    """gateway side 字符串 -> SilverQuant order_type 整数。"""
    text = str(side or "").strip().upper()
    if text in ("SELL", "24", xtconstant.CREDIT_SELL):
        return xtconstant.STOCK_SELL
    return xtconstant.STOCK_BUY


def _order_type_to_side(order_type: Any) -> str:
    """SilverQuant order_type -> gateway side 字符串。"""
    try:
        value = int(order_type)
    except (TypeError, ValueError):
        value = xtconstant.STOCK_BUY
    return "SELL" if value == xtconstant.STOCK_SELL else "BUY"


def _resolve_account_id(account: Any) -> str:
    """从 StockAccount 或字符串中取 account_id。"""
    if account is None:
        return ""
    if isinstance(account, str):
        return account
    return str(getattr(account, "account_id", "") or "")


def _resolve_account_type(account: Any) -> str:
    if account is None:
        return "STOCK"
    return str(getattr(account, "account_type", "STOCK") or "STOCK")


class XtQuantTrader:
    """与大 QMT helper HTTP 网关交互的交易代理。"""

    def __init__(self, path: str = "", session_id: int = 0) -> None:
        # path 与 session_id 在 miniqmt 中用于本地 IPC，这里仅保留以兼容构造签名
        self.path = path
        self.session_id = session_id
        self._callback: Optional[XtQuantTraderCallback] = None
        self._connected = False
        self._subscribed = False
        self._cb_thread: Optional[threading.Thread] = None
        self._cb_stop = threading.Event()
        self._seen_order_ids: set = set()
        self._seen_trade_ids: set = set()
        self._last_orders: List[Dict[str, Any]] = []
        self._last_trades: List[Dict[str, Any]] = []
        from ._bridge import load_config
        self._cb_interval = float(load_config().get("callback_poll_interval_seconds", 3))

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def register_callback(self, callback: XtQuantTraderCallback) -> None:
        self._callback = callback

    def start(self) -> None:
        # miniqmt 在此启动后台通信线程；shim 无需额外线程
        pass

    def connect(self) -> int:
        """连接网关，返回 0 表示成功。"""
        try:
            get_client().health()
            self._connected = True
            return 0
        except Exception as exc:
            _LOGGER.error("connect failed: %s", exc)
            self._connected = False
            return -1

    def disconnect(self) -> None:
        self._connected = False

    def subscribe(self, account: Any) -> int:
        """订阅交易主推，返回 0 表示成功。"""
        if not self._connected:
            _LOGGER.warning("subscribe called before successful connect")
            return -1
        self._start_callback_polling(account)
        self._subscribed = True
        return 0

    def stop(self) -> None:
        self._cb_stop.set()
        if self._cb_thread is not None and self._cb_thread.is_alive():
            self._cb_thread.join(timeout=self._cb_interval + 2)
        self._cb_thread = None
        self._connected = False
        self._subscribed = False

    # ------------------------------------------------------------------
    # 下单 / 撤单
    # ------------------------------------------------------------------

    def _resolve_market_price(
        self,
        security: str,
        price: float,
        order_type: int,
        passorder_pr_type: int,
    ) -> float:
        """市价单保护价：优先用最新价±溢价；否则传 0 让 passorder 取涨跌停。"""
        if price and price > 0:
            return float(price)

        from ._bridge import load_config
        premium = float(load_config().get("market_order_premium", 0.02))
        is_buy = _order_type_to_side(order_type) == "BUY"

        try:
            ticks = get_client().current_tick([security])
            tick = ticks.get(security) if ticks else None
            if tick:
                last = tick.get("lastPrice") or tick.get("last_price")
                if last and float(last) > 0:
                    last_f = float(last)
                    if is_buy:
                        return round(last_f * (1 + premium), 2)
                    return round(last_f * (1 - premium), 2)
        except Exception as exc:
            _LOGGER.debug("resolve market price from tick failed: %s", exc)

        # 大 QMT passorder：市价单 price=0 时自动取涨跌停作为保护限价
        if int(passorder_pr_type) in _PASSORDER_MARKET_PR_TYPES:
            return 0.0
        return 0.01

    def _place_order_via_gateway(
        self,
        account: Any,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str,
        order_remark: str,
    ) -> Dict[str, Any]:
        client = get_client()
        account_id = _resolve_account_id(account)
        account_type = _resolve_account_type(account)
        side = _order_type_to_side(order_type)
        passorder_pr_type = _to_passorder_pr_type(price_type)
        is_market = _is_market_price_type(price_type)
        effective_price = (
            self._resolve_market_price(stock_code, float(price), order_type, passorder_pr_type)
            if is_market
            else float(price)
        )
        _LOGGER.info(
            "place_order %s %s vol=%s miniqmt_pr=%s passorder_pr=%s price=%s",
            stock_code, side, order_volume, price_type, passorder_pr_type, effective_price,
        )
        result = client.place_order(
            security=stock_code,
            side=side,
            amount=int(order_volume),
            price=effective_price,
            account_id=account_id,
            account_type=account_type,
            strategy_name=str(strategy_name or ""),
            order_remark=str(order_remark or ""),
            pr_type=int(passorder_pr_type),
            market=is_market,
        )
        return result

    def order_stock(
        self,
        account: Any,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str,
        order_remark: str,
    ) -> int:
        """同步下单，返回 0 表示提交成功。"""
        try:
            result = self._place_order_via_gateway(
                account, stock_code, order_type, order_volume,
                price_type, price, strategy_name, order_remark,
            )
            order_id = str(result.get("order_id") or "")
            warning = str(result.get("warning") or result.get("error_msg") or "")
            if warning:
                _LOGGER.warning("order_stock response: %s", warning)
            if self._callback is not None:
                resp = XtOrderResponse(
                    order_id=order_id,
                    order_remark=str(order_remark or ""),
                    error_msg=warning,
                )
                try:
                    self._callback.on_order_stock_async_response(resp)
                except Exception as exc:
                    _LOGGER.warning("on_order_stock_async_response failed: %s", exc)
            return 0 if order_id or result.get("status") not in ("rejected",) else -1
        except GatewayError as exc:
            _LOGGER.error("order_stock failed: %s", exc)
            if self._callback is not None:
                try:
                    from .xttype import XtOrderError
                    self._callback.on_order_error(
                        XtOrderError(order_remark=str(order_remark or ""), error_msg=str(exc))
                    )
                except Exception:
                    pass
            return -1

    def order_stock_async(
        self,
        account: Any,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str,
        order_remark: str,
    ) -> int:
        """异步下单，立即返回 0；结果通过回调推送。"""
        try:
            result = self._place_order_via_gateway(
                account, stock_code, order_type, order_volume,
                price_type, price, strategy_name, order_remark,
            )
            order_id = str(result.get("order_id") or "")
            if self._callback is not None:
                resp = XtOrderResponse(
                    order_id=order_id,
                    order_remark=str(order_remark or ""),
                    error_msg=str(result.get("error_msg") or result.get("warning") or ""),
                )
                try:
                    self._callback.on_order_stock_async_response(resp)
                except Exception as exc:
                    _LOGGER.warning("on_order_stock_async_response failed: %s", exc)
            return 0
        except GatewayError as exc:
            _LOGGER.error("order_stock_async failed: %s", exc)
            if self._callback is not None:
                try:
                    from .xttype import XtOrderError
                    self._callback.on_order_error(
                        XtOrderError(order_remark=str(order_remark or ""), error_msg=str(exc))
                    )
                except Exception:
                    pass
            return -1

    def cancel_order_stock(self, account: Any, order_id: Any) -> int:
        """同步撤单，返回 0 表示成功。"""
        try:
            get_client().cancel_order(
                order_id=str(order_id),
                account_id=_resolve_account_id(account),
                account_type=_resolve_account_type(account),
            )
            return 0
        except GatewayError as exc:
            _LOGGER.error("cancel_order_stock failed: %s", exc)
            return -1

    def cancel_order_stock_async(self, account: Any, order_id: Any) -> int:
        """异步撤单，返回 0；结果通过回调推送。"""
        result_code = self.cancel_order_stock(account, order_id)
        if self._callback is not None:
            try:
                resp = XtCancelOrderResponse(
                    order_id=str(order_id),
                    cancel_result=result_code,
                )
                self._callback.on_cancel_order_stock_async_response(resp)
            except Exception as exc:
                _LOGGER.warning("on_cancel_order_stock_async_response failed: %s", exc)
        return result_code

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query_stock_asset(self, account: Any) -> XtAsset:
        account_id = _resolve_account_id(account)
        account_type = _resolve_account_type(account)
        data = get_client().account(account_id, account_type)
        return build_asset(account_id, account_type, data)

    def query_stock_positions(self, account: Any) -> List[XtPosition]:
        account_id = _resolve_account_id(account)
        rows = get_client().positions(account_id, _resolve_account_type(account))
        return [build_position(account_id, row) for row in (rows or [])]

    def query_stock_orders(self, account: Any, cancelable_only: bool = False) -> List[XtOrder]:
        account_id = _resolve_account_id(account)
        rows = get_client().orders(account_id, _resolve_account_type(account))
        orders = [build_order(account_id, row, _side_to_order_type(row.get("side"))) for row in (rows or [])]
        if cancelable_only:
            return [o for o in orders if o.price_type not in [
                xtconstant.BROKER_PRICE_PROP_SUBSCRIBE,
                xtconstant.BROKER_PRICE_PROP_FUND_ENTRUST,
                xtconstant.BROKER_PRICE_PROP_ETF,
                xtconstant.BROKER_PRICE_PROP_DEBT_CONVERSION,
            ]]
        return orders

    def query_stock_order(self, account: Any, order_id: Any) -> Optional[XtOrder]:
        account_id = _resolve_account_id(account)
        rows = get_client().orders(account_id, _resolve_account_type(account), order_id=str(order_id))
        if not rows:
            return None
        return build_order(account_id, rows[0], _side_to_order_type(rows[0].get("side")))

    def query_stock_trades(self, account: Any) -> List[XtTrade]:
        account_id = _resolve_account_id(account)
        rows = get_client().trades(account_id, _resolve_account_type(account))
        return [build_trade(account_id, row, _side_to_order_type(row.get("side"))) for row in (rows or [])]

    def query_ipo_data(self) -> Dict[str, Any]:
        # helper 未提供 IPO 数据接口，返回空 dict 保持兼容
        return {}

    def query_new_purchase_limit(self, account: Any) -> Dict[str, Any]:
        # helper 未提供新股额度查询接口，返回空 dict 保持兼容
        return {}

    # ------------------------------------------------------------------
    # 回调轮询线程
    # ------------------------------------------------------------------

    def _start_callback_polling(self, account: Any) -> None:
        if self._cb_thread is not None and self._cb_thread.is_alive():
            return
        self._cb_stop.clear()
        self._seen_order_ids.clear()
        self._seen_trade_ids.clear()
        self._cb_thread = threading.Thread(
            target=self._callback_loop, args=(account,), name="xt-trader-callback", daemon=True
        )
        self._cb_thread.start()

    def _callback_loop(self, account: Any) -> None:
        account_id = _resolve_account_id(account)
        account_type = _resolve_account_type(account)
        client = get_client()
        while not self._cb_stop.is_set():
            try:
                self._poll_orders(client, account_id, account_type)
            except Exception as exc:
                _LOGGER.debug("callback poll orders failed: %s", exc)
            try:
                self._poll_trades(client, account_id, account_type)
            except Exception as exc:
                _LOGGER.debug("callback poll trades failed: %s", exc)
            self._cb_stop.wait(self._cb_interval)

    def _poll_orders(self, client: GatewayClient, account_id: str, account_type: str) -> None:
        if self._callback is None:
            return
        rows = client.orders(account_id, account_type)
        if not rows:
            return
        for row in rows:
            order_id = str(row.get("order_id") or "")
            if not order_id or order_id in self._seen_order_ids:
                continue
            self._seen_order_ids.add(order_id)
            order = build_order(account_id, row, _side_to_order_type(row.get("side")))
            try:
                self._callback.on_stock_order(order)
            except Exception as exc:
                _LOGGER.warning("on_stock_order callback failed: %s", exc)

    def _poll_trades(self, client: GatewayClient, account_id: str, account_type: str) -> None:
        if self._callback is None:
            return
        rows = client.trades(account_id, account_type)
        if not rows:
            return
        for row in rows:
            trade_id = str(row.get("trade_id") or row.get("order_id") or "")
            if not trade_id or trade_id in self._seen_trade_ids:
                continue
            self._seen_trade_ids.add(trade_id)
            trade = build_trade(account_id, row, _side_to_order_type(row.get("side")))
            try:
                self._callback.on_stock_trade(trade)
            except Exception as exc:
                _LOGGER.warning("on_stock_trade callback failed: %s", exc)


# XtQuantTraderCallback 由 xttype 提供，这里再导出一次兼容
# `from xtquant.xttrader import XtQuantTraderCallback` 用法
__all__ = [
    "XtQuantTrader",
    "XtQuantTraderCallback",
]
