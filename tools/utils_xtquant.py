# -*- coding: utf-8 -*-
"""
统一 xtquant 入口：根据 credentials.USE_BIG_QMT 选择包。

- USE_BIG_QMT=False：原生 miniqmt 的 xtquant（券商安装目录）
- USE_BIG_QMT=True ：项目内 btquant（大 QMT HTTP 桥接，用法与 xtquant 同名 API）

DailyHistoryXT、utils_remote_xt 历史日线下载等仅适用于原生 xtquant，
请使用 load_native_xtquant()，在 USE_BIG_QMT=True 时会提示并返回 None。
"""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # 供 PyCharm 静态解析；运行时仍由 _bind_package 按 USE_BIG_QMT 绑定
    from btquant import xtconstant as xtconstant
    from btquant import xtdata as xtdata
    from btquant import xttrader as xttrader
    from btquant import xttype as xttype
    from btquant.xtconstant import (
        FIX_PRICE as FIX_PRICE,
        LATEST_PRICE as LATEST_PRICE,
        MARKET_PEER_PRICE_FIRST as MARKET_PEER_PRICE_FIRST,
        MARKET_SZ_CONVERT_5_CANCEL as MARKET_SZ_CONVERT_5_CANCEL,
        STOCK_BUY as STOCK_BUY,
        STOCK_SELL as STOCK_SELL,
    )
    from btquant.xttrader import XtQuantTrader as XtQuantTrader, XtQuantTraderCallback as XtQuantTraderCallback
    from btquant.xttype import (
        StockAccount as StockAccount,
        XtAsset as XtAsset,
        XtCancelError as XtCancelError,
        XtCancelOrderResponse as XtCancelOrderResponse,
        XtOrder as XtOrder,
        XtOrderError as XtOrderError,
        XtOrderResponse as XtOrderResponse,
        XtPosition as XtPosition,
        XtTrade as XtTrade,
    )

_logger = logging.getLogger(__name__)

NATIVE_ONLY_HINT = (
    "DailyHistoryXT / utils_remote_xt 历史日线下载仅支持 miniqmt 原生 xtquant；"
    "USE_BIG_QMT=True（大 QMT 桥接 btquant）时不适用，请改用 Tushare/MOOTDX 或关闭 USE_BIG_QMT。"
)


def use_big_qmt() -> bool:
    try:
        from credentials import USE_BIG_QMT
        return bool(USE_BIG_QMT)
    except ImportError:
        return False


def load_xtquant() -> Any:
    """交易/行情主路径。"""
    if use_big_qmt():
        import btquant as xtquant
        return xtquant
    try:
        import xtquant
    except ImportError as exc:
        raise ImportError(
            "未找到 miniqmt 原生 xtquant；请确认 QMT 已安装并配置 PYTHONPATH，"
            "或在 credentials.py 设置 USE_BIG_QMT=True 使用 btquant 大 QMT 桥接。"
        ) from exc
    return xtquant


def load_native_xtquant(warn: bool = True) -> Optional[Any]:
    """仅 miniqmt 原生 xtquant；大 QMT 模式下不可用。"""
    if use_big_qmt():
        if warn:
            _logger.warning(NATIVE_ONLY_HINT)
            print(f"[WARN] {NATIVE_ONLY_HINT}")
        return None
    import xtquant
    return xtquant


def warn_native_only(feature: str = "该功能") -> None:
    if use_big_qmt():
        print(f"[WARN] {feature} {NATIVE_ONLY_HINT}")


def _import_native_submodules() -> tuple[Any, Any, Any, Any]:
    """pip 版 xtquant 250807+ 顶层 __init__ 不再挂载 xtdata 等子模块，需显式 import。"""
    _xtdata = importlib.import_module('xtquant.xtdata')
    _xtconstant = importlib.import_module('xtquant.xtconstant')
    _xttrader = importlib.import_module('xtquant.xttrader')
    _xttype = importlib.import_module('xtquant.xttype')
    return _xtdata, _xtconstant, _xttrader, _xttype


def _bind_package(_pkg: Any) -> None:
    """把包内子模块绑定到 loader 模块级变量。"""
    global xtquant, xtdata, xtconstant, xttrader, xttype
    global XtQuantTrader, XtQuantTraderCallback
    global StockAccount, XtAsset, XtPosition, XtOrder, XtTrade
    global XtOrderError, XtCancelError, XtOrderResponse, XtCancelOrderResponse
    global STOCK_BUY, STOCK_SELL

    xtquant = _pkg
    if use_big_qmt():
        from btquant import xtdata as _xtdata
        from btquant import xtconstant as _xtconstant
        from btquant import xttrader as _xttrader
        from btquant import xttype as _xttype
    else:
        _xtdata, _xtconstant, _xttrader, _xttype = _import_native_submodules()

    xtdata = _xtdata
    xtconstant = _xtconstant
    xttrader = _xttrader
    xttype = _xttype

    XtQuantTrader = _xttrader.XtQuantTrader
    XtQuantTraderCallback = _xttrader.XtQuantTraderCallback
    StockAccount = _xttype.StockAccount
    XtAsset = _xttype.XtAsset
    XtPosition = _xttype.XtPosition
    XtOrder = _xttype.XtOrder
    XtTrade = _xttype.XtTrade
    XtOrderError = _xttype.XtOrderError
    XtCancelError = _xttype.XtCancelError
    XtOrderResponse = _xttype.XtOrderResponse
    XtCancelOrderResponse = _xttype.XtCancelOrderResponse
    STOCK_BUY = _xtconstant.STOCK_BUY
    STOCK_SELL = _xtconstant.STOCK_SELL

    global FIX_PRICE, LATEST_PRICE
    global MARKET_PEER_PRICE_FIRST, MARKET_SZ_CONVERT_5_CANCEL
    FIX_PRICE = _xtconstant.FIX_PRICE
    LATEST_PRICE = _xtconstant.LATEST_PRICE
    MARKET_PEER_PRICE_FIRST = _xtconstant.MARKET_PEER_PRICE_FIRST
    MARKET_SZ_CONVERT_5_CANCEL = _xtconstant.MARKET_SZ_CONVERT_5_CANCEL


def resolve_market_order(code: str, reference_price: float) -> tuple[int, float]:
    """市价单的 price_type / price，与 delegate/xt_delegate.py order_market_open 一致。

    常量定义：
      USE_BIG_QMT=True  -> btquant/xtconstant.py
      USE_BIG_QMT=False -> 券商 xtquant.xtconstant
    """
    from tools.utils_basic import get_code_exchange

    exchange = get_code_exchange(code)
    if exchange == 'SZ':
        return MARKET_SZ_CONVERT_5_CANCEL, -1
    if exchange == 'SH':
        return MARKET_PEER_PRICE_FIRST, reference_price
    return LATEST_PRICE, reference_price


_bind_package(load_xtquant())

__all__ = [
    "use_big_qmt",
    "load_xtquant",
    "load_native_xtquant",
    "warn_native_only",
    "resolve_market_order",
    "xtquant",
    "xtdata",
    "xtconstant",
    "xttrader",
    "xttype",
    "XtQuantTrader",
    "XtQuantTraderCallback",
    "StockAccount",
    "XtAsset",
    "XtPosition",
    "XtOrder",
    "XtTrade",
    "XtOrderError",
    "XtCancelError",
    "XtOrderResponse",
    "XtCancelOrderResponse",
    "STOCK_BUY",
    "STOCK_SELL",
    "FIX_PRICE",
    "LATEST_PRICE",
    "MARKET_PEER_PRICE_FIRST",
    "MARKET_SZ_CONVERT_5_CANCEL",
]
