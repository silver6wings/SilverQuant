# -*- coding: utf-8 -*-
"""
xtquant 影子包内部共享工具：
  - 读取SilverQuant根目录下的 qmt_bridge.json
  - 封装到 big_qmt helper (默认 127.0.0.1:9000) 的 HTTP/JSON 调用
  - miniqmt 风格代码 (000001.SH/000001.SZ) 与 helper 返回的聚宽风格
    (000001.XSHG/000001.XSHE) 互转
  - 行情快照字段归一化

设计原则：不修改SilverQuant任何现有文件，仅靠本影子包把 xtquant 调用代理到大 QMT。
"""
import json
import os
import threading
import time
import logging
from typing import Any, Dict, List, Optional

import requests

_LOGGER = logging.getLogger("xtquant_bridge")
if not _LOGGER.handlers:
    _LOGGER.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "gateway_url": "http://127.0.0.1:9000",
    "gateway_password": "change_me_gateway_password",
    "gateway_secret": "",
    "account_id": "",
    "account_type": "STOCK",
    "request_timeout_seconds": 15,
    "tick_poll_interval_seconds": 1,
    "callback_poll_interval_seconds": 3,
    "market_order_premium": 0.02,
    "enable_trade_callbacks": True,
    "enable_http_log": False,
}

_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_LOCK = threading.Lock()


def _resolve_config_path() -> str:
    """定位SilverQuant根目录下的 qmt_bridge.json。

    优先级：
      1. 环境变量 QMT_BRIDGE_CONFIG 指定的绝对路径
      2. xtquant 包所在目录的上一级（SilverQuant根目录）
      3. 当前工作目录
    """
    env_path = os.environ.get("QMT_BRIDGE_CONFIG")
    if env_path and os.path.isfile(env_path):
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(os.path.dirname(here), "qmt_bridge.json")
    if os.path.isfile(candidate):
        return candidate
    return os.path.join(os.getcwd(), "qmt_bridge.json")


def load_config() -> Dict[str, Any]:
    """加载并缓存 qmt_bridge.json，环境变量可覆盖任意字段。"""
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    with _CONFIG_LOCK:
        if _CONFIG is not None:
            return _CONFIG
        cfg = dict(_DEFAULT_CONFIG)
        path = _resolve_config_path()
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fp:
                    file_cfg = json.load(fp)
                if isinstance(file_cfg, dict):
                    for key, value in file_cfg.items():
                        if not key.startswith("_"):
                            cfg[key] = value
                _LOGGER.info("xtquant bridge config loaded from %s", path)
            else:
                _LOGGER.warning("qmt_bridge.json not found at %s, using defaults", path)
        except Exception as exc:
            _LOGGER.warning("load qmt_bridge.json failed: %s, using defaults", exc)
        # 环境变量覆盖
        env_map = {
            "gateway_url": "QMT_GATEWAY_URL",
            "gateway_password": "QMT_GATEWAY_PASSWORD",
            "gateway_secret": "QMT_GATEWAY_SECRET",
            "account_id": "QMT_ACCOUNT_ID",
            "account_type": "QMT_ACCOUNT_TYPE",
        }
        for cfg_key, env_key in env_map.items():
            env_value = os.environ.get(env_key)
            if env_value:
                cfg[cfg_key] = env_value
        _CONFIG = cfg
        return cfg


def reload_config() -> Dict[str, Any]:
    """强制重新加载配置（测试或运行期切换账号时用）。"""
    global _CONFIG
    with _CONFIG_LOCK:
        _CONFIG = None
    return load_config()


# ---------------------------------------------------------------------------
# 代码格式互转
# ---------------------------------------------------------------------------

def to_qmt_code(code: Any) -> str:
    """SilverQuant风格 000001.SH / 000001.SZ -> helper 使用的 000001.SH / 000001.SZ。

    big_qmt helper 的 _to_qmt_security 直接接受 .SH/.SZ 后缀，原样返回即可。
    缺省按代码首位推断交易所。
    """
    text = str(code or "").strip()
    if not text:
        return text
    if "." in text:
        return text.upper()
    if text.startswith(("5", "6", "7", "9", "0")):
        # 6/5/9 开头多为沪市，0/3 开头多为深市；这里仅做兜底
        if text.startswith(("6", "5", "9", "7")):
            return text + ".SH"
        return text + ".SZ"
    return text


def from_jq_code(code: Any) -> str:
    """helper 返回的聚宽风格 000001.XSHG / 000001.XSHE -> SilverQuant风格 .SH/.SZ。"""
    text = str(code or "").strip()
    if not text or "." not in text:
        return text
    body, suffix = text.rsplit(".", 1)
    suffix = suffix.upper()
    if suffix in ("XSHG", "SSE", "SS"):
        return body + ".SH"
    if suffix in ("XSHE", "SZE", "SZSE"):
        return body + ".SZ"
    return text


# ---------------------------------------------------------------------------
# 行情快照归一化
# ---------------------------------------------------------------------------

# helper 的 _enrich_tick 可能因 _basic_value 丢弃列表字段，这里补齐 5 档默认值，
# 保证SilverQuant record_tick_to_memory / qmt_quote_to_tick 不会 KeyError。
_TICK_ARRAY_FIELDS = ["askPrice", "askVol", "bidPrice", "bidVol"]


def normalize_tick(tick: Dict[str, Any]) -> Dict[str, Any]:
    """把 helper 返回的 tick 归一化成SilverQuant期望的 xtdata 快照结构。"""
    if not isinstance(tick, dict):
        return {}
    item = dict(tick)
    # last_price -> lastPrice（SilverQuant用 camelCase）
    for src, dst in (("last_price", "lastPrice"), ("lastPrice", "lastPrice"), ("price", "lastPrice")):
        value = item.get(src)
        if value not in (None, ""):
            item.setdefault("lastPrice", value)
            break
    for src, dst in (("last_close", "lastClose"), ("lastClose", "lastClose")):
        value = item.get(src)
        if value not in (None, ""):
            item.setdefault("lastClose", value)
            break
    # 5 档兜底
    for field in _TICK_ARRAY_FIELDS:
        value = item.get(field)
        if not isinstance(value, list):
            item[field] = [0, 0, 0, 0, 0]
        else:
            item[field] = (value + [0] * 5)[:5]
    # 时间兜底：helper 可能给 timetag/dt，SilverQuant要 ms 整数 time
    if not item.get("time"):
        timetag = item.get("timetag") or item.get("dt") or item.get("time")
        if timetag:
            try:
                # timetag 形如 "20260808 14:56:30" 或纯数字
                text = str(timetag).replace("-", "").replace(":", "").replace("T", "").replace(" ", "")
                digits = "".join(ch for ch in text if ch.isdigit())
                if len(digits) >= 14:
                    struct = time.strptime(digits[:14], "%Y%m%d%H%M%S")
                    item["time"] = int(time.mktime(struct) * 1000)
                elif len(digits) >= 8:
                    struct = time.strptime(digits[:8], "%Y%m%d")
                    item["time"] = int(time.mktime(struct) * 1000)
            except Exception:
                item["time"] = int(time.time() * 1000)
        else:
            item["time"] = int(time.time() * 1000)
    return item


# ---------------------------------------------------------------------------
# HTTP 网关客户端
# ---------------------------------------------------------------------------

class GatewayError(RuntimeError):
    pass


class GatewayClient:
    """对 big_qmt helper HTTP 网关的薄封装。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.base_url = str(self.config.get("gateway_url", "http://127.0.0.1:9000")).rstrip("/")
        self.password = str(self.config.get("gateway_password", ""))
        self.secret = str(self.config.get("gateway_secret", ""))
        self.timeout = float(self.config.get("request_timeout_seconds", 15))
        self.enable_log = bool(self.config.get("enable_http_log", False))
        self._session = requests.Session()

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if self.password:
            headers["X-BulletTrade-Password"] = self.password
        if self.secret and not self.secret.startswith("change_me"):
            headers["X-BulletTrade-Secret"] = self.secret
        return headers

    def _log(self, route: str, payload: Any, resp: Any, elapsed: float) -> None:
        if not self.enable_log:
            return
        try:
            _LOGGER.info("gateway %s payload=%s ok=%s elapsed=%.0fms", route, payload, resp.get("ok"), elapsed * 1000)
        except Exception:
            pass

    def request(self, route: str, payload: Optional[Dict[str, Any]] = None, method: str = "POST") -> Dict[str, Any]:
        """发起请求，返回解析后的 JSON dict。

        helper 约定：成功返回 {"ok": true, "value": ...}，失败返回 {"ok": false, "code":..., "message":...}。
        本方法在 ok=true 时直接返回 value（便于调用方使用），ok=false 抛 GatewayError。
        /health 比较特殊，直接返回 {"ok": true, "value": {...}}。
        """
        payload = dict(payload or {})
        url = self.base_url + route
        headers = self._headers()
        started = time.time()
        try:
            if method.upper() == "GET":
                response = self._session.get(url, headers=headers, params=payload, timeout=self.timeout)
            else:
                response = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
            elapsed = time.time() - started
            try:
                body = response.json()
            except Exception:
                raise GatewayError("gateway %s returned non-json: %s" % (route, response.text[:200]))
        except requests.RequestException as exc:
            raise GatewayError("gateway %s request failed: %s" % (route, exc))
        self._log(route, payload, body, elapsed)
        if not isinstance(body, dict):
            raise GatewayError("gateway %s returned non-dict: %r" % (route, body))
        if not body.get("ok"):
            raise GatewayError("gateway %s failed: %s %s" % (route, body.get("code"), body.get("message")))
        return body.get("value") if "value" in body else body

    # 便捷封装 ---------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self.request("/health", {}, method="GET")

    def current_tick(self, securities: List[str]) -> Dict[str, Any]:
        """返回 {SilverQuantcode: tick_dict}。"""
        payload = {"securities": [to_qmt_code(c) for c in securities]}
        value = self.request("/data/current_tick", payload)
        ticks = value.get("ticks") if isinstance(value, dict) else None
        if not isinstance(ticks, dict):
            return {}
        result: Dict[str, Any] = {}
        for key, tick in ticks.items():
            code = from_jq_code(key)
            if code:
                result[code] = normalize_tick(tick if isinstance(tick, dict) else {})
        return result

    def history(self, security: str, frequency: str = "1d", start: str = "", end: str = "",
                count: int = -1, fq: str = "none", fields: Optional[List[str]] = None,
                auto_download: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "security": to_qmt_code(security),
            "frequency": frequency,
            "start": start,
            "end": end,
            "fq": fq,
            "auto_download": auto_download,
        }
        if fields is not None:
            payload["fields"] = list(fields)
        if count and count > 0:
            payload["count"] = count
        return self.request("/data/history", payload)

    def trade_days(self, start: str = "", end: str = "", count: int = 250) -> List[str]:
        payload = {"start": start, "end": end, "count": count}
        value = self.request("/data/trade_days", payload)
        values = value.get("values") if isinstance(value, dict) else None
        return list(values or [])

    def account(self, account_id: str = "", account_type: str = "") -> Dict[str, Any]:
        payload = {"account_id": account_id, "account_type": account_type}
        return self.request("/account", payload)

    def positions(self, account_id: str = "", account_type: str = "") -> List[Dict[str, Any]]:
        payload = {"account_id": account_id, "account_type": account_type}
        value = self.request("/positions", payload)
        return value.get("positions") if isinstance(value, dict) else (value or [])

    def orders(self, account_id: str = "", account_type: str = "", order_id: str = "") -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"account_id": account_id, "account_type": account_type}
        if order_id:
            payload["order_id"] = order_id
        value = self.request("/orders", payload)
        return value.get("orders") if isinstance(value, dict) else (value or [])

    def trades(self, account_id: str = "", account_type: str = "", order_id: str = "") -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"account_id": account_id, "account_type": account_type}
        if order_id:
            payload["order_id"] = order_id
        value = self.request("/trades", payload)
        return value.get("trades") if isinstance(value, dict) else (value or [])

    def place_order(self, security: str, side: str, amount: int, price: float,
                    account_id: str = "", account_type: str = "",
                    strategy_name: str = "", order_remark: str = "",
                    pr_type: int = 11, market: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "security": to_qmt_code(security),
            "side": side,
            "amount": int(amount),
            "price": float(price),
            "account_id": account_id,
            "account_type": account_type,
            "strategy_name": strategy_name,
            "order_remark": order_remark,
            "pr_type": int(pr_type),
            "style": {"type": "market" if market else "limit", "price": float(price)},
        }
        return self.request("/place_order", payload)

    def cancel_order(self, order_id: str, account_id: str = "", account_type: str = "") -> Dict[str, Any]:
        payload = {"order_id": order_id, "account_id": account_id, "account_type": account_type}
        return self.request("/cancel_order", payload)


# 单例（懒加载）
_CLIENT: Optional[GatewayClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client() -> GatewayClient:
    global _CLIENT
    if _CLIENT is None:
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = GatewayClient()
    return _CLIENT
