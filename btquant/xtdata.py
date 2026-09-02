# -*- coding: utf-8 -*-
"""
xtquant.xtdata 影子实现。

把SilverQuant对 miniqmt xtdata 的调用代理到大 QMT helper HTTP 网关：
  - get_full_tick           -> /data/current_tick
  - subscribe_whole_quote   -> 后台线程轮询 /data/current_tick，回调推送
  - unsubscribe_quote       -> 停止轮询线程
  - download_history_data   -> /data/ensure_cache
  - get_market_data         -> 多次 /data/history，聚合成 {field: DataFrame(code x dates)}
  - get_client              -> 返回模拟 client，is_connected / down_all_sector_data
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from ._bridge import GatewayClient, GatewayError, get_client as _get_bridge_client, from_jq_code, to_qmt_code

_LOGGER = logging.getLogger("xtquant.xtdata")

# SilverQuant会设置 xtdata.enable_hello = False 来关闭日志
enable_hello: bool = True


# ---------------------------------------------------------------------------
# 模拟 client 对象
# ---------------------------------------------------------------------------

class _XtDataClient:
    """模拟 xtdata.get_client() 返回的连接对象。"""

    def __init__(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        try:
            _get_bridge_client().health()
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def down_all_sector_data(self) -> None:
        # 大 QMT 板块数据下载，helper 暂未提供独立路由；静默忽略
        _LOGGER.debug("down_all_sector_data skipped (no gateway route)")

    def download_sector_data(self) -> None:
        self.down_all_sector_data()


_CLIENT_INSTANCE: Optional[_XtDataClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> _XtDataClient:
    """返回单例 client 对象，兼容 xtdata.get_client()。"""
    global _CLIENT_INSTANCE
    if _CLIENT_INSTANCE is None:
        with _CLIENT_LOCK:
            if _CLIENT_INSTANCE is None:
                _CLIENT_INSTANCE = _XtDataClient()
    return _CLIENT_INSTANCE


# ---------------------------------------------------------------------------
# 全推行情订阅
# ---------------------------------------------------------------------------

class _Subscription:
    """一个订阅会话：后台线程定期拉取 tick 并回调。"""

    def __init__(self, seq: int, codes: List[str], callback: Callable[[Dict[str, Any]], None],
                 interval: float) -> None:
        self.seq = seq
        self.codes = list(codes)
        self.callback = callback
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"xtdata-sub-{self.seq}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                quotes = _get_bridge_client().current_tick(self.codes)
                if quotes:
                    self.callback(quotes)
            except Exception as exc:
                _LOGGER.warning("subscribe_whole_quote poll failed: %s", exc)
            self._stop_event.wait(self.interval)


_SUB_SEQ = 0
_SUB_LOCK = threading.Lock()
_SUBSCRIPTIONS: Dict[int, _Subscription] = {}


def subscribe_whole_quote(code_list: List[str], callback: Callable[[Dict[str, Any]], None]) -> int:
    """订阅全推行情，返回订阅序列号。

    大 QMT helper 是 HTTP 拉取模型，这里用后台线程按配置间隔轮询
    /data/current_tick，把结果通过 callback 推回SilverQuant。
    """
    global _SUB_SEQ
    codes = [to_qmt_code(c) for c in (code_list or [])]
    with _SUB_LOCK:
        _SUB_SEQ += 1
        seq = _SUB_SEQ
        from ._bridge import load_config
        interval = float(load_config().get("tick_poll_interval_seconds", 1))
        sub = _Subscription(seq, codes, callback, interval)
        _SUBSCRIPTIONS[seq] = sub
    sub.start()
    return seq


def unsubscribe_quote(seq: int) -> None:
    """取消订阅。"""
    with _SUB_LOCK:
        sub = _SUBSCRIPTIONS.pop(seq, None)
    if sub is not None:
        sub.stop()


def run() -> None:
    """阻塞主线程，保持订阅后台线程运行（兼容 miniqmt 的 xtdata.run）。"""
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# 实时快照
# ---------------------------------------------------------------------------

def get_full_tick(code_list: List[str]) -> Dict[str, Any]:
    """获取全推快照，返回 {SilverQuantcode: tick_dict}。"""
    if not code_list:
        return {}
    return _get_bridge_client().current_tick([to_qmt_code(c) for c in code_list])


# 兼容别名
def get_full_tick2(code_list: List[str]) -> Dict[str, Any]:
    return get_full_tick(code_list)


# ---------------------------------------------------------------------------
# 历史数据下载
# ---------------------------------------------------------------------------

def download_history_data(code: str, period: str = '1d', start_time: str = '',
                           end_time: str = '', incrementally: Any = None) -> int:
    """触发大 QMT 下载历史数据到本地缓存，返回 0 表示成功。"""
    client = _get_bridge_client()
    payload: Dict[str, Any] = {
        "security": to_qmt_code(code),
        "frequency": period,
        "start": start_time,
        "end": end_time,
    }
    if incrementally is not None:
        payload["incrementally"] = incrementally
    try:
        client.request("/data/ensure_cache", payload)
        return 0
    except GatewayError as exc:
        _LOGGER.warning("download_history_data failed for %s: %s", code, exc)
        return -1


# ---------------------------------------------------------------------------
# 历史数据读取
# ---------------------------------------------------------------------------

def _dividend_type(adjust: str) -> str:
    """SilverQuant复权标识 -> helper 的 fq 字段。"""
    text = str(adjust or "none").strip().lower()
    mapping = {
        "front": "qfq",
        "back": "hfq",
        "none": "none",
        "front_ratio": "qfq",
        "back_ratio": "hfq",
        "follow": "none",
        "qfq": "qfq",
        "hfq": "hfq",
        "bfq": "none",
        "": "none",
    }
    return mapping.get(text, "none")


def _ts_to_date_str(ts: Any) -> str:
    """把 QMT history 里的 time 字段转成 'YYYYMMDD' 日期字符串。"""
    if ts in (None, ""):
        return ""
    try:
        value = float(ts)
    except (TypeError, ValueError):
        text = str(ts).replace("-", "").replace(":", "").replace("T", "").replace(" ", "")
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits[:8]
    # QMT 时间戳通常是毫秒
    if value > 1e12:
        value = value / 1000.0
    return time.strftime("%Y%m%d", time.localtime(value))


def _history_records_to_field_dict(records: List[List[Any]], columns: List[str],
                                    code: str) -> Dict[str, Any]:
    """把 gateway 的 dataframe payload 转成 {field: value}，按日期对齐。

    返回 {field_name: [(date_str, value), ...]} 的中间结构。
    """
    if not records or not columns:
        return {}
    time_idx = None
    for idx, col in enumerate(columns):
        if str(col).lower() in ("time", "datetime", "date", "timetag"):
            time_idx = idx
            break
    result: Dict[str, List] = {}
    dates: List[str] = []
    for record in records:
        if time_idx is not None and time_idx < len(record):
            dates.append(_ts_to_date_str(record[time_idx]))
        else:
            dates.append("")
    for col_idx, col_name in enumerate(columns):
        key = str(col_name).lower()
        if key in ("time", "datetime", "date", "timetag"):
            continue
        values = []
        for row_idx, record in enumerate(records):
            if col_idx < len(record):
                values.append((dates[row_idx], record[col_idx]))
            else:
                values.append((dates[row_idx], None))
        result[str(col_name)] = values
    return result


def get_market_data(
    field_list: Optional[List[str]] = None,
    stock_list: Optional[List[str]] = None,
    period: str = '1d',
    start_time: str = '',
    end_time: str = '',
    count: int = -1,
    dividend_type: str = 'none',
    fill_data: bool = False,
) -> Dict[str, pd.DataFrame]:
    """获取历史行情数据。

    返回与 miniqmt xtdata.get_market_data 一致的结构：
        {field_name: DataFrame(index=stock_codes, columns=dates)}
    """
    if stock_list is None:
        stock_list = []
    if field_list is None:
        field_list = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount']

    client = _get_bridge_client()
    fq = _dividend_type(dividend_type)
    fields = [f for f in field_list if f and f.lower() != "time"]
    # 保留 time 字段用于提取日期
    request_fields = list(dict.fromkeys(["time"] + fields)) if "time" not in fields else list(field_list)

    # code -> {field: [(date, value), ...]}
    code_field_map: Dict[str, Dict[str, List]] = {}
    all_dates: set = set()

    for raw_code in stock_list:
        qmt_code = to_qmt_code(raw_code)
        try:
            value = client.history(
                security=qmt_code,
                frequency=period,
                start=start_time,
                end=end_time,
                fq=fq,
                fields=request_fields,
            )
        except GatewayError as exc:
            _LOGGER.warning("get_market_data history failed for %s: %s", qmt_code, exc)
            continue
        columns = value.get("columns") if isinstance(value, dict) else None
        records = value.get("records") if isinstance(value, dict) else None
        if not columns or not records:
            continue
        field_data = _history_records_to_field_dict(records, [str(c) for c in columns], qmt_code)
        if field_data:
            code_field_map[raw_code] = field_data
            for field_values in field_data.values():
                for date_str, _ in field_values:
                    if date_str:
                        all_dates.add(date_str)

    sorted_dates = sorted(all_dates)

    # 构建 {field: DataFrame(index=codes, columns=dates)}
    result: Dict[str, pd.DataFrame] = {}
    if not sorted_dates:
        for field in fields:
            result[field] = pd.DataFrame(index=stock_list, columns=[])
        return result

    for field in fields:
        rows = {}
        for raw_code in stock_list:
            field_data = code_field_map.get(raw_code)
            if not field_data:
                rows[raw_code] = [None] * len(sorted_dates)
                continue
            pairs = field_data.get(field)
            if not pairs:
                rows[raw_code] = [None] * len(sorted_dates)
                continue
            date_to_value = {d: v for d, v in pairs if d}
            rows[raw_code] = [date_to_value.get(d) for d in sorted_dates]
        result[field] = pd.DataFrame.from_dict(rows, orient="index", columns=sorted_dates)

    return result


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------

def get_trading_dates_by_market(market: str = 'SH', start_time: str = '', end_time: str = '',
                                 count: int = 250) -> List[str]:
    """获取交易日列表。"""
    return _get_bridge_client().trade_days(start=start_time, end=end_time, count=count)


# ---------------------------------------------------------------------------
# 兼容方法（SilverQuant未直接调用但 xtquant 标准接口存在）
# ---------------------------------------------------------------------------

def get_instrument_detail(code: str) -> Dict[str, Any]:
    """获取标的详情，兼容 xtdata.get_instrument_detail。"""
    try:
        return _get_bridge_client().request("/data/security_info", {"security": to_qmt_code(code)})
    except GatewayError:
        return {}


def get_stock_list_in_sector(sector_name: str) -> List[str]:
    """获取板块成分股，兼容 xtdata.get_stock_list_in_sector。"""
    try:
        value = _get_bridge_client().request("/data/all_securities", {"sector": sector_name})
        records = value.get("records") if isinstance(value, dict) else None
        columns = value.get("columns") if isinstance(value, dict) else None
        if not records or not columns:
            return []
        # 第 0 列是 jq 风格代码
        return [from_jq_code(row[0]) for row in records if row]
    except GatewayError:
        return []
