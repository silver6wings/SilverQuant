# -*- coding: utf-8 -*-
"""
xtquant.xttype 影子实现。

提供与 xttype 同名的数据类与回调基类，让SilverQuant的 delegate/callback 代码无需修改即可
基于大 QMT helper 返回的 dict 构造出对象。
"""
from typing import Any, Dict, Optional

from ._bridge import from_jq_code


class _Base:
    """通用属性容器：构造时把 kwargs 写成属性，未知属性默认 None。"""

    __slots__ = ()

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:  # pragma: no cover - 调试辅助
        pairs = ", ".join("%s=%r" % (k, getattr(self, k, None)) for k in self.__slots__)
        return "%s(%s)" % (self.__class__.__name__, pairs)


class StockAccount(_Base):
    __slots__ = ("account_id", "account_type")

    def __init__(self, account_id: str = "", account_type: str = "STOCK"):
        super().__init__(account_id=account_id, account_type=account_type)


class XtAsset(_Base):
    __slots__ = ("account_id", "account_type", "cash", "available_cash",
                 "frozen_cash", "market_value", "total_asset")

    def __init__(self, account_id: str = "", account_type: str = "STOCK", cash: float = 0.0,
                 available_cash: float = 0.0, frozen_cash: float = 0.0,
                 market_value: float = 0.0, total_asset: float = 0.0):
        super().__init__(account_id=account_id, account_type=account_type, cash=cash,
                         available_cash=available_cash, frozen_cash=frozen_cash,
                         market_value=market_value, total_asset=total_asset)


class XtPosition(_Base):
    __slots__ = ("account_id", "stock_code", "volume", "can_use_volume",
                 "frozen_volume", "open_price", "cost", "market_value",
                 "last_price", "name")

    def __init__(self, account_id: str = "", stock_code: str = "", volume: int = 0,
                 can_use_volume: int = 0, frozen_volume: int = 0,
                 open_price: float = 0.0, cost: float = 0.0,
                 market_value: float = 0.0, last_price: float = 0.0, name: str = ""):
        super().__init__(account_id=account_id, stock_code=stock_code, volume=volume,
                         can_use_volume=can_use_volume, frozen_volume=frozen_volume,
                         open_price=open_price, cost=cost, market_value=market_value,
                         last_price=last_price, name=name)


class XtOrder(_Base):
    __slots__ = ("account_id", "order_id", "order_sysid", "stock_code",
                 "order_type", "order_volume", "price", "price_type",
                 "order_status", "order_remark", "traded_volume",
                 "strategy_name", "order_time")

    def __init__(self, account_id: str = "", order_id: str = "", order_sysid: str = "",
                 stock_code: str = "", order_type: Optional[int] = None,
                 order_volume: int = 0, price: float = 0.0, price_type: int = 0,
                 order_status: int = 0, order_remark: str = "",
                 traded_volume: int = 0, strategy_name: str = "", order_time: int = 0):
        super().__init__(account_id=account_id, order_id=order_id, order_sysid=order_sysid,
                         stock_code=stock_code, order_type=order_type, order_volume=order_volume,
                         price=price, price_type=price_type, order_status=order_status,
                         order_remark=order_remark, traded_volume=traded_volume,
                         strategy_name=strategy_name, order_time=order_time)


class XtTrade(_Base):
    __slots__ = ("account_id", "trade_id", "order_id", "order_sysid",
                 "stock_code", "order_type", "traded_volume", "traded_price",
                 "traded_time", "order_remark", "strategy_name")

    def __init__(self, account_id: str = "", trade_id: str = "", order_id: str = "",
                 order_sysid: str = "", stock_code: str = "", order_type: Optional[int] = None,
                 traded_volume: int = 0, traded_price: float = 0.0,
                 traded_time: int = 0, order_remark: str = "", strategy_name: str = ""):
        super().__init__(account_id=account_id, trade_id=trade_id, order_id=order_id,
                         order_sysid=order_sysid, stock_code=stock_code, order_type=order_type,
                         traded_volume=traded_volume, traded_price=traded_price,
                         traded_time=traded_time, order_remark=order_remark,
                         strategy_name=strategy_name)


class XtOrderError(_Base):
    __slots__ = ("order_id", "error_id", "error_msg", "order_remark")

    def __init__(self, order_id: str = "", error_id: int = 0,
                 error_msg: str = "", order_remark: str = ""):
        super().__init__(order_id=order_id, error_id=error_id,
                         error_msg=error_msg, order_remark=order_remark)


class XtCancelError(_Base):
    __slots__ = ("order_id", "error_id", "error_msg")

    def __init__(self, order_id: str = "", error_id: int = 0, error_msg: str = ""):
        super().__init__(order_id=order_id, error_id=error_id, error_msg=error_msg)


class XtOrderResponse(_Base):
    __slots__ = ("order_id", "order_remark", "error_msg")

    def __init__(self, order_id: str = "", order_remark: str = "", error_msg: str = ""):
        super().__init__(order_id=order_id, order_remark=order_remark, error_msg=error_msg)


class XtCancelOrderResponse(_Base):
    __slots__ = ("order_id", "order_sysid", "error_msg", "cancel_result")

    def __init__(self, order_id: str = "", order_sysid: str = "",
                 error_msg: str = "", cancel_result: int = 0):
        super().__init__(order_id=order_id, order_sysid=order_sysid,
                         error_msg=error_msg, cancel_result=cancel_result)


class XtAccountStatus(_Base):
    __slots__ = ("account_id", "account_type", "status")

    def __init__(self, account_id: str = "", account_type: str = "", status: int = 0):
        super().__init__(account_id=account_id, account_type=account_type, status=status)


class XtQuantTraderCallback:
    """与 xtquant.xttrader.XtQuantTraderCallback 同名的回调基类。

    子类按需覆盖 on_* 方法。代理 trader 会在收到 helper 数据/订单/成交变化时调用对应方法。
    """

    def on_disconnected(self):
        pass

    def on_stock_trade(self, trade: XtTrade):
        pass

    def on_stock_order(self, order: XtOrder):
        pass

    def on_order_stock_async_response(self, response: XtOrderResponse):
        pass

    def on_order_error(self, order_error: XtOrderError):
        pass

    def on_cancel_order_stock_async_response(self, response: XtCancelOrderResponse):
        pass

    def on_cancel_error(self, cancel_error: XtCancelError):
        pass

    def on_account_status(self, status: XtAccountStatus):
        pass

    def on_stock_position(self, position: XtPosition):
        pass


# ---------------------------------------------------------------------------
# 从 helper dict 构造数据类的工厂方法
# ---------------------------------------------------------------------------

def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def build_asset(account_id: str, account_type: str, data: Dict[str, Any]) -> XtAsset:
    return XtAsset(
        account_id=account_id,
        account_type=account_type,
        cash=_float(data.get("cash") or data.get("available_cash")),
        available_cash=_float(data.get("available_cash")),
        frozen_cash=_float(data.get("frozen_cash")),
        market_value=_float(data.get("market_value")),
        total_asset=_float(data.get("total_value") or data.get("total_asset")),
    )


def build_position(account_id: str, data: Dict[str, Any]) -> XtPosition:
    return XtPosition(
        account_id=account_id,
        stock_code=from_jq_code(str(data.get("security") or data.get("stock_code") or "")),
        volume=_int(data.get("amount")),
        can_use_volume=_int(data.get("closeable_amount") or data.get("can_use_volume")),
        frozen_volume=_int(data.get("frozen")),
        open_price=_float(data.get("avg_cost") or data.get("cost_basis") or data.get("open_price")),
        cost=_float(data.get("cost_basis") or data.get("avg_cost")),
        market_value=_float(data.get("market_value")),
        last_price=_float(data.get("last_price")),
        name=str(data.get("name") or ""),
    )


def build_order(account_id: str, data: Dict[str, Any], order_type: Optional[int] = None) -> XtOrder:
    return XtOrder(
        account_id=account_id,
        order_id=str(data.get("order_id") or ""),
        order_sysid=str(data.get("order_sysid") or ""),
        stock_code=from_jq_code(str(data.get("security") or data.get("stock_code") or "")),
        order_type=order_type if order_type is not None else _int(data.get("order_type"), 0) or None,
        order_volume=_int(data.get("amount")),
        price=_float(data.get("price") or data.get("order_price")),
        price_type=_int(data.get("price_type")),
        order_status=_int(data.get("raw_status") or data.get("order_status")),
        order_remark=str(data.get("order_remark") or data.get("remark") or ""),
        traded_volume=_int(data.get("filled")),
        strategy_name=str(data.get("strategy_name") or ""),
        order_time=_int(data.get("order_time")),
    )


def build_trade(account_id: str, data: Dict[str, Any], order_type: Optional[int] = None) -> XtTrade:
    return XtTrade(
        account_id=account_id,
        trade_id=str(data.get("trade_id") or ""),
        order_id=str(data.get("order_id") or ""),
        order_sysid=str(data.get("order_sysid") or ""),
        stock_code=from_jq_code(str(data.get("security") or data.get("stock_code") or "")),
        order_type=order_type if order_type is not None else _int(data.get("order_type"), 0) or None,
        traded_volume=_int(data.get("amount")),
        traded_price=_float(data.get("price")),
        traded_time=_int(data.get("trade_time") or data.get("time")),
        order_remark=str(data.get("order_remark") or data.get("remark") or ""),
        strategy_name=str(data.get("strategy_name") or ""),
    )
