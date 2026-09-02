#encoding:gbk
# Author: BruceLee
# Date: 2026-07-02
# Version: 20260703_miniqmt_alignment
# File: Big QMT embedded gateway strategy sample.
# Description: Run inside a dedicated Big QMT strategy and expose a
# BulletTrade-compatible local HTTP/JSON data and trading gateway.
# Big QMT embedded gateway strategy sample.
# Run this file inside a dedicated Big QMT strategy to expose a local HTTP/JSON
# bridge. Validate scheduling, thread boundaries and QMT API availability in a
# simulation environment before production use.

import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import builtins
import hashlib
from typing import Any, Dict, List, Optional
from uuid import uuid4

import tornado.ioloop
import tornado.web


# =========================
# Operator Configuration
# =========================

# Local bind host for the HTTP gateway. Keep 127.0.0.1 unless another machine
# must connect directly to this QMT process.
LISTEN_HOST = "127.0.0.1"

# Local bind port for the HTTP gateway. The BulletTrade side must use the same
# port in BIG_QMT_GATEWAY_URL or related configuration.
LISTEN_PORT = 9000

# Build marker shown in startup logs and /health. Update this when copying a new
# helper build into QMT so tests can prove the running file version.
GATEWAY_BUILD_ID = "20260703_miniqmt_alignment"

# Shared password required by non-health HTTP APIs. Change this to a private
# local value outside simulation; clients send it as X-BulletTrade-Password or
# Authorization: Bearer <password>.
GATEWAY_PASSWORD = "change_me_gateway_password"

# Optional extra secret header for stronger local auth. If this no longer starts
# with change_me, clients must send the same X-BulletTrade-Secret value.
GATEWAY_SECRET = "change_me_hmac_secret"

# QMT fund account id. Requests may override it with account_id, but setting it
# here is the normal gateway mode. Keep placeholder in shared examples.
ACCOUNT_ID = "8881234567"

# QMT account type used by get_trade_detail_data/passorder. Stock accounts use
# stock; other account types must match the broker/QMT environment.
ACCOUNT_TYPE = "stock"

# Security used when callers ask for trade days without passing a symbol.
# MiniQMT accepts symbol-less trade-day calls; Big QMT needs a concrete symbol,
# so use a stable liquid A-share as the equivalent market calendar anchor.
DEFAULT_TRADE_DAYS_SECURITY = "000001.SZ"

# Match MiniQMT's safety default: history reads first request QMT to prepare
# local cache, then read local bars. Callers may pass auto_download=false only
# for explicit diagnostics.
AUTO_ENSURE_HISTORY_CACHE = True

# When automatic history cache preparation fails, return an error instead of
# reading potentially stale or placeholder local data.
HISTORY_FAIL_ON_ENSURE_CACHE_ERROR = True

# Match MiniQMT's index constituent semantics: prefer index-weight APIs over
# sector lists. Sector list is only a fallback when the broker build lacks the
# index-weight functions.
INDEX_WEIGHT_DOWNLOAD_BEFORE_READ = True

# Allow /place_order to call QMT passorder. Set False for data-only or
# read-only production verification.
ENABLE_TRADING = True

# Allow /cancel_order to call QMT cancel APIs for one order id.
ENABLE_CANCEL_ORDER = True

# Allow future bulk cancel routes. Keep False unless bulk cancel has been
# explicitly tested in the target QMT environment.
ENABLE_CANCEL_ALL = False

# Allow safer rule-based cancel helpers that first inspect known orders.
ENABLE_RULE_CANCEL = True

# Include limited raw trade-detail debug data in logs/responses when useful for
# mapping QMT object fields. Disable in production if logs should be smaller.
ENABLE_TRADE_DETAIL_DEBUG = True

# Maximum raw rows sampled for trade-detail debug output.
TRADE_DETAIL_DEBUG_MAX_ROWS = 3

# Virtual sub accounts can share one real QMT account. The gateway stores the
# virtual owner in the order remark so later order/trade queries can split rows.
# Prefix written into order remarks/tags for virtual sub-account ownership.
VIRTUAL_ACCOUNT_REMARK_PREFIX = "sub:"

# Maximum full BulletTrade remark length stored by the helper. This is separate
# from QMT's official userOrderId field, which has a much shorter limit.
ORDER_REMARK_MAX_LENGTH = 64

# Maximum userOrderId length sent into Big QMT passorder. Official docs describe
# this as the investment remark/user-defined id and require length < 24.
QMT_USER_ORDER_ID_MAX_LENGTH = 23

# Local JSON file name for mapping QMT order ids/refs to virtual account tags.
ORDER_TAG_STORE_FILE_NAME = "bt_big_qmt_order_tags.json"

# How long local order tag mappings are kept before pruning.
ORDER_TAG_MATCH_TTL_SECONDS = 24 * 60 * 60

# Maximum number of local order tag mappings kept in memory/file.
MAX_ORDER_TAGS = 5000

# Call passorder with the official userOrderId/investment-remark argument first.
# If the broker build only supports the article-compatible signature, the helper
# catches TypeError and falls back automatically.
PASSORDER_USE_REMARK_SIGNATURE = True

# passorder quickTrade argument. The article sample uses 2. If passorder returns
# 0 and no order appears, set this to 0 or 1 in QMT for broker-specific testing.
PASSORDER_QUICK_TRADE = 2

# Match the article sample's passorder semantics: a zero/empty return means QMT
# did not give us a usable order id, but it is not proof that the request was
# rejected. Return submit_unknown so the BulletTrade side can poll orders before
# deciding whether the submission really failed.
PASSORDER_EMPTY_RETURN_AS_SUBMIT_UNKNOWN = True

# Allow authenticated requests to override passorder quickTrade with quick_trade
# or quickTrade. Keep enabled in simulation diagnostics; disable in production if
# the gateway must force one broker-verified quickTrade mode.
ALLOW_PASSORDER_QUICK_TRADE_OVERRIDE = True

# Match the verified article sample: start Tornado from init(ContextInfo) on the
# current Big QMT strategy thread. Background-thread mode can accept requests in
# some QMT builds but may fail to flush HTTP responses to the client.
# Start the HTTP server in a Python background thread when True. Keep False for
# the verified QMT strategy lifecycle mode unless diagnosing run-mode behavior.
RUN_HTTP_IN_BACKGROUND_THREAD = False

# QMT may call stop() when a run task finishes. Keep False for gateway service
# mode so an early stop callback does not kill the local HTTP gateway.
# Stop Tornado when QMT calls stop(). Gateway service mode normally keeps this
# False so the port remains alive after early stop callbacks.
STOP_HTTP_ON_QMT_STOP = False

# Formal gateway mode should run from the Big QMT strategy lifecycle, where QMT
# calls init(ContextInfo). Enable this only for diagnostics when checking whether
# a run mode merely executes module-level code.
# Start HTTP at module import time instead of waiting for init(ContextInfo).
# Diagnostic only; most data/trade APIs still require ContextInfo.
AUTO_START_HTTP_ON_MODULE_LOAD = False

# Default per-request wait time when dispatching queued QMT actions.
REQUEST_TIMEOUT_SECONDS = 10

# Maximum accepted HTTP JSON body size in bytes.
MAX_REQUEST_BODY_BYTES = 1024 * 1024

# Maximum queued QMT action count before the gateway rejects new work.
MAX_QUEUE_SIZE = 100

# These APIs are the same family as the article sample: they use the Big QMT
# global get_trade_detail_data function and do not need ContextInfo or
# handlebar scheduling.
# Dispatch account/position/order/trade reads directly through Big QMT globals
# when possible instead of queueing through the strategy bar callback.
DIRECT_DISPATCH_ACCOUNT_APIS = True

# Python logger level for console/file logs.
LOG_LEVEL = logging.INFO

# Optional log directory. Empty means QMT script directory when __file__ exists,
# otherwise current working directory.
LOG_DIR = ""

# Log file name created under LOG_DIR or the resolved default directory.
LOG_FILE_NAME = "bt_big_qmt_gateway.log"


def _boot_print(message: str) -> None:
    try:
        print("[BT_BIG_QMT] %s" % message)
    except Exception:
        pass


_boot_print("module loaded build_id=%s; waiting for QMT init(ContextInfo)" % GATEWAY_BUILD_ID)


def _resolve_log_file() -> str:
    if LOG_DIR:
        base_dir = LOG_DIR
    else:
        script_file = globals().get("__file__", "")
        if script_file:
            base_dir = os.path.dirname(os.path.abspath(script_file))
        else:
            base_dir = os.getcwd()
    if not base_dir:
        base_dir = os.getcwd()
    try:
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    except Exception:
        base_dir = os.getcwd()
    return os.path.abspath(os.path.join(base_dir, LOG_FILE_NAME))


try:
    LOG_FILE = _resolve_log_file()
    _boot_print("log file resolved log_file=%s cwd=%s file=%s" % (LOG_FILE, os.getcwd(), globals().get("__file__", "<no __file__>")))
except Exception as exc:
    LOG_FILE = os.path.abspath(LOG_FILE_NAME)
    _boot_print("log file resolve failed fallback=%s error=%s" % (LOG_FILE, exc))


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("bt_big_qmt_gateway")
    logger.setLevel(LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_exists = False
    for handler in logger.handlers:
        if getattr(handler, "baseFilename", None) == LOG_FILE:
            file_exists = True
            handler.setLevel(LOG_LEVEL)
            handler.setFormatter(formatter)
    if not file_exists:
        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        except TypeError:
            file_handler = logging.FileHandler(LOG_FILE)
        except Exception as exc:
            _boot_print("file logger disabled log_file=%s error=%s" % (LOG_FILE, exc))
            file_handler = None
        if file_handler is not None:
            file_handler.setLevel(LOG_LEVEL)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    logger.propagate = True
    return logger


LOGGER = _setup_logger()
_boot_print("logger initialized")


def _emit(level: str, message: str, *args: Any) -> None:
    try:
        text = message % args if args else message
    except Exception:
        text = message
    line = "[BT_BIG_QMT] %s" % text
    try:
        print(line)
    except Exception:
        pass
    try:
        if level == "error":
            LOGGER.error(text)
        elif level == "warning":
            LOGGER.warning(text)
        else:
            LOGGER.info(text)
    except Exception:
        pass


_emit(
    "info",
    "logger ready log_file=%s cwd=%s file=%s",
    LOG_FILE,
    os.getcwd(),
    globals().get("__file__", "<no __file__>"),
)

ORDER_TAG_LOCK = threading.Lock()
ORDER_TAG_STORE_LOADED = False
ORDER_TAGS_BY_ID: Dict[str, Dict[str, Any]] = {}
PENDING_ORDER_TAGS: List[Dict[str, Any]] = []


def _ok(value: Any, request_id: Optional[str] = None) -> Dict[str, Any]:
    return {"ok": True, "value": value, "request_id": request_id, "ts": time.time()}


def _error(code: str, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "ok": False,
        "code": code,
        "message": message,
        "request_id": request_id,
        "ts": time.time(),
    }


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _check_password(headers: Any) -> bool:
    password = (
        headers.get("X-BulletTrade-Password")
        or headers.get("X-BT-Gateway-Password")
        or ""
    )
    secret = headers.get("X-BulletTrade-Secret") or headers.get("X-BT-Gateway-Secret") or ""
    auth = headers.get("Authorization") or ""
    if auth.startswith("Bearer "):
        password = password or auth.split(" ", 1)[1].strip()
    password_ok = bool(GATEWAY_PASSWORD) and password == GATEWAY_PASSWORD
    if not password_ok:
        return False
    if GATEWAY_SECRET and not str(GATEWAY_SECRET).startswith("change_me"):
        return secret == GATEWAY_SECRET
    return True


def _basic_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    try:
        if hasattr(value, "isoformat"):
            return value.isoformat()
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return [_basic_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _basic_value(item) for key, item in value.items()}
    result = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if callable(item):
            continue
        if isinstance(item, (str, int, float, bool, type(None))):
            result[name] = item
    return result or str(value)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _payload_bool(payload: Dict[str, Any], keys: List[str], default: bool) -> bool:
    for key in keys:
        if key in payload:
            return _to_bool(payload.get(key), default)
    return default


def _security(code: Any, exchange: Any) -> str:
    text = str(code or "")
    suffix = str(exchange or "").upper()
    if "." in text:
        return _from_qmt_security(text)
    if suffix in ("SH", "SSE", "XSHG"):
        return "%s.XSHG" % text
    if suffix in ("SZ", "SZE", "XSHE"):
        return "%s.XSHE" % text
    return "%s.%s" % (text, suffix) if suffix else text


def _to_qmt_security(security: Any) -> str:
    text = str(security or "").strip()
    if not text:
        return text
    if "." in text:
        code, suffix = text.rsplit(".", 1)
        suffix = suffix.upper()
        if suffix in ("XSHE", "SZE", "SZSE"):
            return "%s.SZ" % code
        if suffix in ("XSHG", "SSE", "SS"):
            return "%s.SH" % code
        return "%s.%s" % (code, suffix)
    if text.startswith(("5", "6", "7", "9")):
        return "%s.SH" % text
    if text.startswith(("0", "1", "2", "3")):
        return "%s.SZ" % text
    return text


def _from_qmt_security(security: Any) -> str:
    text = str(security or "").strip()
    if not text or "." not in text:
        return text
    code, suffix = text.rsplit(".", 1)
    suffix = suffix.upper()
    if suffix in ("SZ", "SZE", "XSHE", "SZSE"):
        return "%s.XSHE" % code
    if suffix in ("SH", "SS", "SSE", "XSHG"):
        return "%s.XSHG" % code
    return "%s.%s" % (code, suffix)


def _qmt_security_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [item.strip() for item in values.split(",") if item.strip()]
    result = []
    for item in values:
        qmt_code = _to_qmt_security(item)
        if qmt_code:
            result.append(qmt_code)
    return result


def _tick_float(tick: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = tick.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _enrich_tick(context_info: Any, qmt_security: str, tick: Any) -> Dict[str, Any]:
    item = _basic_value(tick)
    if not isinstance(item, dict):
        item = {"raw": item}
    security = _from_qmt_security(qmt_security)
    item.setdefault("security", security)
    item.setdefault("sid", security)
    last_price = _tick_float(item, "last_price", "lastPrice", "price")
    if last_price is not None:
        item.setdefault("last_price", last_price)
    timetag = item.get("timetag") or item.get("time") or item.get("datetime")
    if timetag not in (None, ""):
        item.setdefault("dt", timetag)
    getter = getattr(context_info, "get_instrument_detail", None)
    if getter is None:
        getter = getattr(context_info, "get_instrumentdetail", None)
    info = {}
    if getter is not None:
        try:
            info = getter(qmt_security, True)
        except TypeError:
            try:
                info = getter(qmt_security)
            except Exception:
                info = {}
        except Exception:
            info = {}
    if isinstance(info, dict):
        high_limit = _tick_float(info, "UpStopPrice", "high_limit", "HighLimit")
        low_limit = _tick_float(info, "DownStopPrice", "low_limit", "LowLimit")
        if high_limit is not None:
            item.setdefault("high_limit", high_limit)
        if low_limit is not None:
            item.setdefault("low_limit", low_limit)
    open_int = item.get("openInt")
    if open_int in (None, ""):
        open_int = item.get("stockStatus")
    if open_int not in (None, ""):
        try:
            item.setdefault("paused", int(open_int) in (1, 17, 20))
        except Exception:
            pass
    return item


def _normalize_tick_keys(ticks: Any, context_info: Any = None) -> Dict[str, Any]:
    if not isinstance(ticks, dict):
        return {}
    normalized = {}
    for key, value in ticks.items():
        normalized[_from_qmt_security(key)] = _enrich_tick(context_info, str(key), value)
    return normalized


def _is_placeholder_account_id(account_id: Any) -> bool:
    text = str(account_id or "").strip()
    return text in ("", "change_me_account_id", "xxxxxxx", "your_account_id")


def _resolve_account(payload: Dict[str, Any]) -> Dict[str, str]:
    account_id = payload.get("account_id") or payload.get("accountID") or ACCOUNT_ID
    account_type = payload.get("account_type") or payload.get("account") or ACCOUNT_TYPE
    return {"account_id": str(account_id or ""), "account_type": str(account_type or ACCOUNT_TYPE)}


def _resolve_virtual_account_id(payload: Dict[str, Any]) -> str:
    value = (
        payload.get("sub_account_id")
        or payload.get("subAccountId")
        or payload.get("virtual_account_id")
        or payload.get("virtualAccountId")
        or payload.get("virtual_account")
        or ""
    )
    text = str(value or "").strip()
    if "@" in text:
        text = text.split("@", 1)[0].strip()
    return text


def _extract_virtual_account_id_from_remark(remark: Any) -> str:
    text = str(remark or "").strip()
    if not text:
        return ""
    separators = ["|", ";", ",", " "]
    tokens = [text]
    for separator in separators:
        next_tokens = []
        for token in tokens:
            next_tokens.extend(token.split(separator))
        tokens = next_tokens
    markers = [
        VIRTUAL_ACCOUNT_REMARK_PREFIX,
        "sub_account_id=",
        "virtual_account_id=",
        "sub=",
        "virtual=",
    ]
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        for marker in markers:
            if token.startswith(marker):
                return token[len(marker):].strip()
    return ""


def _remark_matches_virtual_account(remark: Any, sub_account_id: str) -> bool:
    sub_account_id = str(sub_account_id or "").strip()
    if not sub_account_id:
        return True
    text = str(remark or "").strip()
    if not text:
        return False
    if text == sub_account_id:
        return True
    extracted = _extract_virtual_account_id_from_remark(text)
    if extracted and extracted == sub_account_id:
        return True
    explicit_tokens = [
        VIRTUAL_ACCOUNT_REMARK_PREFIX + sub_account_id,
        "sub_account_id=" + sub_account_id,
        "virtual_account_id=" + sub_account_id,
        "sub=" + sub_account_id,
        "virtual=" + sub_account_id,
    ]
    return any(token in text for token in explicit_tokens)


def _truncate_order_remark(remark: str) -> str:
    max_len = _to_int(ORDER_REMARK_MAX_LENGTH, 0)
    if max_len <= 0 or len(remark) <= max_len:
        return remark
    return remark[:max_len]


def _truncate_qmt_user_order_id(value: str) -> str:
    max_len = _to_int(QMT_USER_ORDER_ID_MAX_LENGTH, 23)
    if max_len <= 0 or len(value) <= max_len:
        return value
    return value[:max_len]


def _compose_qmt_user_order_id(payload: Dict[str, Any], order_remark: Optional[str]) -> str:
    for key in ("qmt_user_order_id", "user_order_id", "userOrderId", "user_orderid", "order_user_id"):
        value = payload.get(key)
        if value not in (None, ""):
            return _truncate_qmt_user_order_id(str(value).strip())
    request_id = str(payload.get("request_id") or "")
    seed = "|".join([str(order_remark or ""), request_id, str(time.time()), uuid4().hex])
    try:
        digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    except Exception:
        digest = uuid4().hex
    return _truncate_qmt_user_order_id("BT" + digest[:20])


def _compose_order_remark(payload: Dict[str, Any]) -> Optional[str]:
    base = payload.get("order_remark")
    if base is None:
        base = payload.get("remark")
    base_text = str(base).strip() if base not in (None, "") else ""
    sub_account_id = _resolve_virtual_account_id(payload)
    if not sub_account_id:
        return _truncate_order_remark(base_text) if base_text else None
    if _remark_matches_virtual_account(base_text, sub_account_id):
        return _truncate_order_remark(base_text)
    prefix = VIRTUAL_ACCOUNT_REMARK_PREFIX + sub_account_id
    if base_text:
        return _truncate_order_remark(prefix + "|" + base_text)
    return _truncate_order_remark(prefix)


def _matches_order_remark_filter(row: Dict[str, Any], remark_filter: Any) -> bool:
    if remark_filter in (None, ""):
        return True
    expected = str(remark_filter)
    actual = str(row.get("order_remark") or row.get("remark") or "")
    return expected in actual


def _filter_virtual_account_rows(rows: List[Dict[str, Any]], payload: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not payload:
        return rows
    sub_account_id = _resolve_virtual_account_id(payload)
    if sub_account_id:
        rows = [
            row
            for row in rows
            if str(row.get("sub_account_id") or row.get("virtual_account_id") or "") == sub_account_id
            or _remark_matches_virtual_account(row.get("order_remark") or row.get("remark"), sub_account_id)
        ]
    remark_filter = payload.get("order_remark_filter") or payload.get("remark_filter")
    if remark_filter is None and payload.get("filter_order_remark"):
        remark_filter = payload.get("order_remark") or payload.get("remark")
    if remark_filter not in (None, ""):
        rows = [row for row in rows if _matches_order_remark_filter(row, remark_filter)]
    strategy_name = payload.get("strategy_name_filter") or payload.get("strategy_filter")
    if strategy_name not in (None, ""):
        rows = [row for row in rows if str(row.get("strategy_name") or row.get("strategy") or "") == str(strategy_name)]
    return rows


def _order_tag_store_path() -> str:
    return os.path.join(os.path.dirname(LOG_FILE), ORDER_TAG_STORE_FILE_NAME)


def _load_order_tag_store_once() -> None:
    global ORDER_TAG_STORE_LOADED, ORDER_TAGS_BY_ID, PENDING_ORDER_TAGS
    if ORDER_TAG_STORE_LOADED:
        return
    with ORDER_TAG_LOCK:
        if ORDER_TAG_STORE_LOADED:
            return
        path = _order_tag_store_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fp:
                    payload = json.load(fp)
                by_id = payload.get("by_id") if isinstance(payload, dict) else {}
                pending = payload.get("pending") if isinstance(payload, dict) else []
                ORDER_TAGS_BY_ID = dict(by_id or {})
                PENDING_ORDER_TAGS = list(pending or [])
        except Exception as exc:
            LOGGER.warning("load order tag store failed: %s", exc)
            ORDER_TAGS_BY_ID = {}
            PENDING_ORDER_TAGS = []
        ORDER_TAG_STORE_LOADED = True


def _save_order_tag_store_locked() -> None:
    try:
        _prune_order_tags_locked()
        path = _order_tag_store_path()
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        payload = {
            "by_id": ORDER_TAGS_BY_ID,
            "pending": PENDING_ORDER_TAGS[-MAX_ORDER_TAGS:],
            "updated_at": time.time(),
        }
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2, sort_keys=True)
            fp.write("\n")
    except Exception as exc:
        LOGGER.warning("save order tag store failed: %s", exc)


def _prune_order_tags_locked() -> None:
    now = time.time()
    ttl = max(60, _to_int(ORDER_TAG_MATCH_TTL_SECONDS, 24 * 60 * 60))
    old_ids = []
    for key, tag in ORDER_TAGS_BY_ID.items():
        created_at = float(tag.get("created_at") or now)
        if now - created_at > ttl:
            old_ids.append(key)
    for key in old_ids:
        ORDER_TAGS_BY_ID.pop(key, None)
    kept = []
    for tag in PENDING_ORDER_TAGS:
        created_at = float(tag.get("created_at") or now)
        if now - created_at <= ttl:
            kept.append(tag)
    del PENDING_ORDER_TAGS[:]
    PENDING_ORDER_TAGS.extend(kept[-MAX_ORDER_TAGS:])


def _remember_order_tag(order_id: Optional[Any], tag: Dict[str, Any]) -> None:
    _load_order_tag_store_once()
    with ORDER_TAG_LOCK:
        clean = dict(tag)
        clean.setdefault("created_at", time.time())
        oid = str(order_id or "").strip()
        if oid and oid != "0":
            ORDER_TAGS_BY_ID[oid] = clean
        PENDING_ORDER_TAGS.append(clean)
        _save_order_tag_store_locked()


def _parse_order_epoch(row: Dict[str, Any]) -> Optional[float]:
    raw = row.get("raw") if isinstance(row, dict) else None
    if not isinstance(raw, dict):
        return None
    date_text = str(raw.get("m_strInsertDate") or "")
    time_text = str(raw.get("m_strInsertTime") or "")
    if not date_text or not time_text:
        return None
    try:
        value = time.strptime(date_text + time_text.zfill(6), "%Y%m%d%H%M%S")
        return time.mktime(value)
    except Exception:
        return None


def _float_close(left: Any, right: Any) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except Exception:
        return False
    return abs(left_value - right_value) <= max(0.01, abs(right_value) * 0.000001)


def _order_matches_tag(row: Dict[str, Any], tag: Dict[str, Any]) -> bool:
    if str(row.get("security") or "") != str(tag.get("security") or ""):
        return False
    amount = int(row.get("amount") or 0)
    tag_amount = int(tag.get("amount") or 0)
    if tag_amount > 0 and amount != tag_amount:
        return False
    tag_price = tag.get("price")
    order_price = row.get("order_price")
    if tag_price not in (None, "") and order_price not in (None, "", 0, 0.0):
        if not _float_close(order_price, tag_price):
            return False
    order_epoch = _parse_order_epoch(row)
    created_at = float(tag.get("created_at") or 0.0)
    if order_epoch is not None and created_at > 0:
        if order_epoch + 300 < created_at:
            return False
    return True


def _apply_order_tag(row: Dict[str, Any], tag: Dict[str, Any]) -> bool:
    changed = False
    qmt_user_order_id = tag.get("qmt_user_order_id")
    if qmt_user_order_id not in (None, "") and not row.get("qmt_user_order_id"):
        row["qmt_user_order_id"] = qmt_user_order_id
        changed = True
    for key in ("order_remark", "remark", "strategy_name", "sub_account_id", "virtual_account_id"):
        value = tag.get(key)
        current = row.get(key)
        should_replace_qmt_id = (
            key in ("order_remark", "remark")
            and qmt_user_order_id not in (None, "")
            and str(current or "") == str(qmt_user_order_id)
        )
        if value not in (None, "") and (not current or should_replace_qmt_id):
            row[key] = value
            changed = True
    return changed


def _attach_virtual_tags_to_orders(orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    _load_order_tag_store_once()
    changed = False
    with ORDER_TAG_LOCK:
        _prune_order_tags_locked()
        for order in orders:
            order_id = str(order.get("order_id") or "").strip()
            tag = ORDER_TAGS_BY_ID.get(order_id) if order_id else None
            if tag is None:
                for pending in reversed(PENDING_ORDER_TAGS):
                    if _order_matches_tag(order, pending):
                        tag = pending
                        if order_id:
                            ORDER_TAGS_BY_ID[order_id] = dict(pending)
                            changed = True
                        break
            if tag is not None and _apply_order_tag(order, tag):
                changed = True
        if changed:
            _save_order_tag_store_locked()
    return orders


def _qmt_period(value: Any) -> str:
    text = str(value or "1d").strip().lower()
    mapping = {
        "daily": "1d",
        "day": "1d",
        "1day": "1d",
        "minute": "1m",
        "min": "1m",
        "1min": "1m",
        "weekly": "1w",
        "month": "1mon",
        "monthly": "1mon",
    }
    return mapping.get(text, text)


def _qmt_dividend_type(value: Any) -> str:
    text = str(value or "follow").strip().lower()
    mapping = {
        "pre": "front_ratio",
        "post": "back_ratio",
        "qfq": "front_ratio",
        "hfq": "back_ratio",
        "none": "none",
        "": "follow",
    }
    return mapping.get(text, text)


def _date_digits(value: Any) -> str:
    text = str(value or "")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 10 and not digits.startswith(("19", "20", "21")):
        try:
            timestamp = int(digits[:13] if len(digits) >= 13 else digits[:10])
            if len(digits) >= 13:
                timestamp = timestamp // 1000
            elif len(digits) > 10:
                timestamp = int(digits)
                timestamp = timestamp // 1000
            return time.strftime("%Y%m%d", time.localtime(timestamp))
        except Exception:
            pass
    return digits[:8]


def _date_in_range(value: Any, start: Any, end: Any) -> bool:
    current = _date_digits(value)
    if not current:
        return True
    start_text = _date_digits(start)
    end_text = _date_digits(end)
    if start_text and current < start_text:
        return False
    if end_text and current > end_text:
        return False
    return True


def _date_iso(value: Any) -> str:
    digits = _date_digits(value)
    if len(digits) == 8:
        return "%s-%s-%s" % (digits[:4], digits[4:6], digits[6:8])
    return str(value or "")


def _is_miniqmt_fund_security(security: Any) -> bool:
    # Keep MiniQMTProvider's current split-dividend convention exactly:
    # 5xxxxx funds/ETFs use per_base=1; other six-digit codes use per_base=10.
    code = str(security or "").split(".", 1)[0]
    return len(code) == 6 and code.startswith("5")


def _split_dividend_row(qmt_security: str, template_security: Any, key: Any, item: Any) -> Dict[str, Any]:
    values = list(item) if isinstance(item, (list, tuple)) else []

    def _number(index: int, default: float = 0.0) -> float:
        try:
            return float(values[index] if index < len(values) else default)
        except Exception:
            return default

    cash_dividend = _number(0)
    bonus_share = _number(1)
    transfer_share = _number(2)
    rights_issue = _number(3)
    is_fund = _is_miniqmt_fund_security(qmt_security)
    per_base = 1 if is_fund else 10
    bonus_pre_tax = cash_dividend if is_fund else cash_dividend * 10.0
    return {
        "security": _from_qmt_security(template_security or qmt_security),
        "date": _date_iso(key),
        "security_type": "fund" if is_fund else "stock",
        "scale_factor": float(1.0 + bonus_share + transfer_share + rights_issue),
        "bonus_pre_tax": float(bonus_pre_tax),
        "per_base": per_base,
    }


def _dataframe_index_values(df: Any) -> List[str]:
    if df is None:
        return []
    try:
        index = getattr(df, "index", None)
        if index is None:
            return []
        try:
            values = list(index.tolist())
        except Exception:
            values = list(index)
        result = []
        for item in values:
            text = _date_digits(item)
            if text:
                result.append(text)
            elif item is not None:
                result.append(str(item))
        return result
    except Exception:
        return []


class QmtApiUnavailable(RuntimeError):
    pass


def _qmt_global(name: str) -> Any:
    value = globals().get(name)
    if value is None:
        value = getattr(builtins, name, None)
    if value is None:
        raise QmtApiUnavailable("QMT API %s is not available" % name)
    return value


def _qmt_global_available(name: str) -> bool:
    return globals().get(name) is not None or getattr(builtins, name, None) is not None


def _context_not_ready(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _error(
        "QMT_CONTEXT_NOT_READY",
        "ContextInfo is not ready; run this file as a Big QMT strategy so init(ContextInfo) is called",
        payload.get("request_id"),
    )


def _dataframe_to_payload(df: Any, include_index: bool = True) -> Dict[str, Any]:
    if df is None:
        return {"dtype": "dataframe", "columns": [], "records": []}
    try:
        columns = [str(col) for col in list(df.columns)]
        has_named_index = bool(getattr(getattr(df, "index", None), "name", None))
        raw = df.reset_index().values.tolist() if include_index and has_named_index else df.values.tolist()
        if include_index and has_named_index:
            columns = [str(df.index.name)] + columns
        return {
            "dtype": "dataframe",
            "columns": columns,
            "records": [[_basic_value(item) for item in row] for row in raw],
        }
    except Exception:
        return {"dtype": "dataframe", "columns": [], "records": []}


def _select_dataframe_columns(df: Any, fields: List[str]) -> Any:
    """Return only requested data columns, matching MiniQMT get_price shape."""
    if df is None or not fields:
        return df
    try:
        available = list(getattr(df, "columns", []))
        selected = [field for field in fields if field in available]
        if selected:
            return df[selected]
    except Exception:
        pass
    return df


def _history_price_decimals(security: Any) -> int:
    code = str(security or "").split(".", 1)[0]
    if len(code) == 6 and code.startswith(("5", "15", "16")):
        return 3
    return 2


def _history_volume_multiplier(period: Any) -> float:
    text = str(period or "").strip().lower()
    if text in ("1m", "1min", "minute", "min"):
        return 100.0
    return 1.0


def _normalize_history_payload(payload: Dict[str, Any], security: Any, period: Any = "") -> Dict[str, Any]:
    price_fields = set(["open", "high", "low", "close", "pre_close", "last_price"])
    columns = payload.get("columns") or []
    records = payload.get("records") or []
    indexes = [idx for idx, column in enumerate(columns) if str(column).lower() in price_fields]
    volume_indexes = [idx for idx, column in enumerate(columns) if str(column).lower() in ("volume", "vol")]
    volume_multiplier = _history_volume_multiplier(period)
    if not indexes and (not volume_indexes or volume_multiplier == 1.0):
        return payload
    if not records:
        return payload
    decimals = _history_price_decimals(security)
    normalized = []
    for row in records:
        values = list(row)
        for idx in indexes:
            if idx >= len(values):
                continue
            value = values[idx]
            if value in (None, ""):
                continue
            try:
                values[idx] = round(float(value), decimals)
            except Exception:
                pass
        if volume_multiplier != 1.0:
            for idx in volume_indexes:
                if idx >= len(values):
                    continue
                value = values[idx]
                if value in (None, ""):
                    continue
                try:
                    values[idx] = float(value) * volume_multiplier
                except Exception:
                    pass
        normalized.append(values)
    result = dict(payload)
    result["records"] = normalized
    return result


def _records_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"dtype": "dataframe", "columns": [], "records": []}
    columns = list(rows[0].keys())
    return {
        "dtype": "dataframe",
        "columns": columns,
        "records": [[row.get(col) for col in columns] for row in rows],
    }


def _account_to_dict(account: Any, account_id: str, account_type: str) -> Dict[str, Any]:
    if account is None:
        return {}
    return {
        "account_id": account_id,
        "account_type": account_type,
        "available_cash": float(getattr(account, "m_dAvailable", 0.0) or 0.0),
        "cash": float(getattr(account, "m_dBalance", 0.0) or 0.0),
        "total_value": float(getattr(account, "m_dBalance", 0.0) or 0.0),
        "market_value": float(getattr(account, "m_dInstrumentValue", 0.0) or 0.0),
        "raw": _basic_value(account),
    }


def _position_to_dict(position: Any) -> Dict[str, Any]:
    return {
        "security": _security(
            getattr(position, "m_strInstrumentID", ""),
            getattr(position, "m_strExchangeID", ""),
        ),
        "name": getattr(position, "m_strInstrumentName", ""),
        "amount": int(getattr(position, "m_nVolume", 0) or 0),
        "closeable_amount": int(getattr(position, "m_nCanUseVolume", 0) or 0),
        "avg_cost": float(getattr(position, "m_dOpenPrice", 0.0) or 0.0),
        "cost_basis": float(getattr(position, "m_dOpenPrice", 0.0) or 0.0),
        "market_value": float(getattr(position, "m_dMarketValue", 0.0) or 0.0),
        "last_price": float(getattr(position, "m_dLastPrice", 0.0) or 0.0),
        "frozen": int(getattr(position, "m_nFrozenVolume", 0) or 0),
        "raw": _basic_value(position),
    }


def _order_to_dict(order: Any) -> Dict[str, Any]:
    raw_status = getattr(order, "m_nOrderStatus", None)
    order_remark = getattr(order, "m_strRemark", None)
    if order_remark in (None, ""):
        order_remark = getattr(order, "m_strUserOrderId", None)
    if order_remark in (None, ""):
        order_remark = getattr(order, "m_strOrderRemark", None)
    qmt_user_order_id = order_remark or ""
    strategy_name = getattr(order, "m_strStrategyName", None)
    virtual_account_id = getattr(order, "sub_account_id", None)
    if virtual_account_id in (None, ""):
        virtual_account_id = getattr(order, "virtual_account_id", None)
    if virtual_account_id in (None, ""):
        virtual_account_id = _extract_virtual_account_id_from_remark(order_remark)
    return {
        "order_id": str(getattr(order, "m_strOrderSysID", "") or ""),
        "security": _security(
            getattr(order, "m_strInstrumentID", ""),
            getattr(order, "m_strExchangeID", ""),
        ),
        "raw_status": raw_status,
        "volume_left": int(getattr(order, "m_nVolumeTotal", 0) or 0),
        "amount": int(getattr(order, "m_nVolumeTotalOriginal", 0) or 0),
        "filled": int(getattr(order, "m_nVolumeTraded", 0) or 0),
        "price": float(getattr(order, "m_dTradedPrice", 0.0) or 0.0),
        "order_price": float(getattr(order, "m_dLimitPrice", 0.0) or 0.0),
        "order_remark": order_remark,
        "remark": order_remark,
        "qmt_user_order_id": qmt_user_order_id,
        "strategy_name": strategy_name,
        "sub_account_id": virtual_account_id or "",
        "virtual_account_id": virtual_account_id or "",
        "raw": _basic_value(order),
    }


def _trade_to_dict(trade: Any) -> Dict[str, Any]:
    price = getattr(trade, "m_dTradePrice", None)
    if price in (None, 0, 0.0):
        price = getattr(trade, "m_dPrice", 0.0)
    order_remark = getattr(trade, "m_strRemark", None)
    if order_remark in (None, ""):
        order_remark = getattr(trade, "m_strUserOrderId", None)
    if order_remark in (None, ""):
        order_remark = getattr(trade, "m_strOrderRemark", None)
    qmt_user_order_id = order_remark or ""
    strategy_name = getattr(trade, "m_strStrategyName", None)
    virtual_account_id = getattr(trade, "sub_account_id", None)
    if virtual_account_id in (None, ""):
        virtual_account_id = getattr(trade, "virtual_account_id", None)
    if virtual_account_id in (None, ""):
        virtual_account_id = _extract_virtual_account_id_from_remark(order_remark)
    return {
        "trade_id": str(getattr(trade, "m_strTradeID", "") or ""),
        "order_id": str(getattr(trade, "m_strOrderSysID", "") or ""),
        "security": _security(
            getattr(trade, "m_strInstrumentID", ""),
            getattr(trade, "m_strExchangeID", ""),
        ),
        "amount": int(getattr(trade, "m_nVolume", 0) or 0),
        "price": float(price or 0.0),
        "order_remark": order_remark,
        "remark": order_remark,
        "qmt_user_order_id": qmt_user_order_id,
        "strategy_name": strategy_name,
        "sub_account_id": virtual_account_id or "",
        "virtual_account_id": virtual_account_id or "",
        "raw": _basic_value(trade),
    }


def _trade_detail(account_id: str, account_type: str, detail_type: str, source: Optional[str] = None) -> List[Any]:
    getter = _qmt_global("get_trade_detail_data")
    try:
        if source:
            return list(getter(account_id, account_type, detail_type, source) or [])
        return list(getter(account_id, account_type, detail_type) or [])
    except Exception as exc:
        LOGGER.exception("get_trade_detail_data failed: type=%s error=%s", detail_type, exc)
        raise


def _try_trade_detail(account_id: str, account_type: str, detail_type: str, source: Optional[str] = None) -> Dict[str, Any]:
    getter = _qmt_global("get_trade_detail_data")
    try:
        if source:
            rows = list(getter(account_id, account_type, detail_type, source) or [])
        else:
            rows = list(getter(account_id, account_type, detail_type) or [])
        return {"ok": True, "rows": rows, "error": None}
    except Exception as exc:
        return {"ok": False, "rows": [], "error": str(exc)}


def _row_identity(row: Any, fallback_prefix: str) -> str:
    names = [
        "m_strOrderSysID",
        "m_strOrderID",
        "m_strOrderRef",
        "m_strTradeID",
        "m_strDealID",
        "m_strInstrumentID",
        "m_strExchangeID",
    ]
    values = []
    for name in names:
        try:
            value = getattr(row, name)
        except Exception:
            value = None
        if value not in (None, ""):
            values.append("%s=%s" % (name, value))
    if values:
        return "|".join(values)
    try:
        return fallback_prefix + ":" + json.dumps(_basic_value(row), ensure_ascii=False, sort_keys=True)
    except Exception:
        return fallback_prefix + ":" + str(row)


def _merge_trade_detail_rows(account_id: str, account_type: str, detail_types: List[str], sources: List[Optional[str]]) -> List[Any]:
    rows = []
    seen = set()
    for detail_type in detail_types:
        for source in sources:
            result = _try_trade_detail(account_id, account_type, detail_type, source)
            for row in result["rows"]:
                key = detail_type + "|" + _row_identity(row, "row")
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def _query_account(account_id: str, account_type: str) -> Dict[str, Any]:
    rows = _trade_detail(account_id, account_type, "account")
    return _account_to_dict(rows[0] if rows else None, account_id, account_type)


def _query_positions(account_id: str, account_type: str) -> List[Dict[str, Any]]:
    return [_position_to_dict(item) for item in _trade_detail(account_id, account_type, "position")]


def _query_orders(account_id: str, account_type: str, payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows = _merge_trade_detail_rows(account_id, account_type, ["order"], ["qmt", None])
    orders = [_order_to_dict(item) for item in rows]
    orders = _attach_virtual_tags_to_orders(orders)
    order_id = payload.get("order_id") if payload else None
    if order_id:
        orders = [item for item in orders if str(item.get("order_id")) == str(order_id)]
    return _filter_virtual_account_rows(orders, payload)


def _query_trades(account_id: str, account_type: str, payload: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    rows = _merge_trade_detail_rows(account_id, account_type, ["deal", "trade"], ["qmt", None])
    trades = [_trade_to_dict(item) for item in rows]
    try:
        orders = _query_orders(account_id, account_type, None)
    except Exception as exc:
        LOGGER.warning("query orders for trade remark enrichment failed: %s", exc)
        orders = []
    orders_by_id = {str(item.get("order_id") or ""): item for item in orders if item.get("order_id")}
    for trade in trades:
        order = orders_by_id.get(str(trade.get("order_id") or ""))
        if not order:
            continue
        trade_remark = trade.get("order_remark")
        order_qmt_user_order_id = order.get("qmt_user_order_id")
        if (
            (not trade_remark)
            or (
                order_qmt_user_order_id not in (None, "")
                and str(trade_remark) == str(order_qmt_user_order_id)
            )
        ) and order.get("order_remark"):
            trade["order_remark"] = order.get("order_remark")
            trade["remark"] = order.get("order_remark")
        if not trade.get("qmt_user_order_id") and order.get("qmt_user_order_id"):
            trade["qmt_user_order_id"] = order.get("qmt_user_order_id")
        if not trade.get("strategy_name") and order.get("strategy_name"):
            trade["strategy_name"] = order.get("strategy_name")
        if not trade.get("sub_account_id") and order.get("sub_account_id"):
            trade["sub_account_id"] = order.get("sub_account_id")
            trade["virtual_account_id"] = order.get("virtual_account_id") or order.get("sub_account_id")
    order_id = payload.get("order_id") if payload else None
    if order_id:
        trades = [item for item in trades if str(item.get("order_id")) == str(order_id)]
    security = payload.get("security") if payload else None
    if security:
        trades = [item for item in trades if str(item.get("security") or "") == str(security)]
    return _filter_virtual_account_rows(trades, payload)


def _get_full_tick(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    securities = payload.get("securities") or payload.get("symbols")
    if not securities:
        securities = payload.get("stocks") or payload.get("codes")
    security = payload.get("security")
    if not securities:
        securities = [security] if security else []
    qmt_codes = _qmt_security_list(securities)
    if context_info is None:
        return _context_not_ready(payload)
    ticks = context_info.get_full_tick(stock_code=qmt_codes) or {}
    source = "ContextInfo.get_full_tick"
    return {"ticks": _normalize_tick_keys(ticks, context_info), "qmt_codes": qmt_codes, "source": source}


def _call_download_history_data(qmt_security: str, period: str, start: Any, end: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    downloader = _qmt_global("download_history_data")
    incrementally = payload.get("incrementally")
    start_text = str(start or "")
    end_text = str(end or "")
    if incrementally is None:
        downloader(qmt_security, period, start_text, end_text)
    else:
        try:
            downloader(qmt_security, period, start_text, end_text, incrementally=_to_bool(incrementally, True))
        except TypeError:
            downloader(qmt_security, period, start_text, end_text)
    return {
        "security": _from_qmt_security(qmt_security),
        "qmt_security": qmt_security,
        "period": period,
        "start": start,
        "end": end,
        "requested": True,
    }


def _auto_ensure_history_cache(qmt_security: str, period: str, start: Any, end: Any, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    enabled = _payload_bool(
        payload,
        ["auto_download", "auto_ensure_cache", "ensure_cache"],
        AUTO_ENSURE_HISTORY_CACHE,
    )
    if not enabled:
        return None
    started = time.time()
    info = _call_download_history_data(qmt_security, period, start, end, payload)
    _emit(
        "info",
        "history auto ensure_cache success security=%s period=%s start=%s end=%s cost_ms=%.1f",
        qmt_security,
        period,
        str(start or ""),
        str(end or ""),
        (time.time() - started) * 1000.0,
    )
    return info


def _query_history(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    security = payload.get("security")
    securities = payload.get("securities") or payload.get("symbols")
    if not security and securities:
        security = securities[0] if isinstance(securities, list) else str(securities).split(",")[0]
    if not security:
        return _error("BAD_REQUEST", "history missing security", payload.get("request_id"))
    qmt_security = _to_qmt_security(security)

    fields = payload.get("fields") or []
    if isinstance(fields, str):
        fields = [item.strip() for item in fields.split(",") if item.strip()]
    period = _qmt_period(payload.get("frequency") or payload.get("period") or "1d")
    start = payload.get("start") or payload.get("start_date") or payload.get("start_time") or ""
    end = payload.get("end") or payload.get("end_date") or payload.get("end_time") or ""
    count = _to_int(payload.get("count"), -1)
    dividend_type = _qmt_dividend_type(payload.get("fq") or payload.get("dividend_type"))
    fill_data = _to_bool(payload.get("fill_data"), _to_bool(payload.get("fill_paused"), True))
    subscribe = _to_bool(payload.get("subscribe"), False)
    try:
        if context_info is None:
            return _context_not_ready(payload)
        try:
            _auto_ensure_history_cache(qmt_security, period, start, end, payload)
        except QmtApiUnavailable as exc:
            LOGGER.exception("history auto ensure_cache unavailable: %s", exc)
            if HISTORY_FAIL_ON_ENSURE_CACHE_ERROR:
                return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
        except Exception as exc:
            LOGGER.exception("history auto ensure_cache failed: %s", exc)
            if HISTORY_FAIL_ON_ENSURE_CACHE_ERROR:
                return _error("ENSURE_CACHE_FAILED", str(exc), payload.get("request_id"))
        data = context_info.get_market_data_ex(
            fields,
            [qmt_security],
            period=period,
            start_time=str(start),
            end_time=str(end),
            count=count,
            dividend_type=dividend_type,
            fill_data=fill_data,
            subscribe=subscribe,
        )
        if isinstance(data, dict):
            df = data.get(qmt_security)
            if df is None and data:
                df = list(data.values())[0]
            df = _select_dataframe_columns(df, fields)
            value = _normalize_history_payload(_dataframe_to_payload(df, include_index=False), security, period)
            return _ok(value, payload.get("request_id"))
        data = _select_dataframe_columns(data, fields)
        value = _normalize_history_payload(_dataframe_to_payload(data, include_index=False), security, period)
        return _ok(value, payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_market_data_ex unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_market_data_ex failed: %s", exc)
        return _error("HISTORY_FAILED", str(exc), payload.get("request_id"))


def _query_trade_days(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    security = payload.get("security") or payload.get("stockcode") or ""
    start = payload.get("start") or payload.get("start_date") or ""
    end = payload.get("end") or payload.get("end_date") or ""
    qmt_start = _date_digits(start)
    qmt_end = _date_digits(end)
    count = _to_int(payload.get("count"), 250)
    period = _qmt_period(payload.get("period") or payload.get("frequency") or "1d")
    try:
        if context_info is None:
            return _context_not_ready(payload)
        qmt_security = _to_qmt_security(security or DEFAULT_TRADE_DAYS_SECURITY)
        values = context_info.get_trading_dates(qmt_security, str(qmt_start), str(qmt_end), count, period) or []
        if not values and period in ("1d", "day", "daily"):
            history = context_info.get_market_data_ex(
                ["close"],
                [qmt_security],
                period="1d",
                start_time=str(qmt_start),
                end_time=str(qmt_end),
                count=count,
                dividend_type="none",
                fill_data=True,
                subscribe=False,
            )
            df = history.get(qmt_security) if isinstance(history, dict) else history
            if df is None and isinstance(history, dict) and history:
                df = list(history.values())[0]
            values = _dataframe_index_values(df)
        return _ok({"dtype": "list", "values": [str(item) for item in values]}, payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_trading_dates unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_trading_dates failed: %s", exc)
        return _error("TRADE_DAYS_FAILED", str(exc), payload.get("request_id"))


def _extract_instrument_type(raw_type: Any) -> Optional[str]:
    if isinstance(raw_type, str):
        cleaned = raw_type.strip().lower()
        return cleaned or None
    if isinstance(raw_type, dict):
        for key, enabled in raw_type.items():
            if enabled:
                cleaned = str(key).strip().lower()
                if cleaned:
                    return cleaned
        if len(raw_type) == 1:
            key = next(iter(raw_type))
            cleaned = str(key).strip().lower()
            return cleaned or None
    return None


def _security_type_from_code(qmt_security: str) -> Optional[str]:
    code = str(qmt_security or "").split(".", 1)[0]
    suffix = str(qmt_security or "").rsplit(".", 1)[-1].upper() if "." in str(qmt_security or "") else ""
    if code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "588", "159")):
        return "etf"
    if code.startswith("399") or (suffix == "SH" and code.startswith("000")):
        return "index"
    return None


def _normalize_security_type(context_info: Any, qmt_security: str, info: Dict[str, Any]) -> str:
    detectors = []
    detector = getattr(context_info, "get_instrument_type", None)
    if callable(detector):
        detectors.append(detector)
    global_detector = globals().get("get_instrument_type") or getattr(builtins, "get_instrument_type", None)
    if callable(global_detector):
        detectors.append(global_detector)
    for detector in detectors:
        try:
            detected = _extract_instrument_type(detector(qmt_security))
        except Exception:
            detected = None
        if detected:
            return detected
    coded = _security_type_from_code(qmt_security)
    if coded:
        return coded
    for key in ("type", "ProductType", "InstrumentType", "SecurityType", "product_type"):
        detected = _extract_instrument_type(info.get(key))
        if detected:
            if "etf" in detected:
                return "etf"
            if "fund" in detected or "\u57fa\u91d1" in detected:
                return "fund"
            if "index" in detected or "\u6307\u6570" in detected:
                return "index"
            return detected
    return "stock"


def _normalize_security_info_value(context_info: Any, security: Any, qmt_security: str, info: Any) -> Dict[str, Any]:
    value = _basic_value(info)
    if not isinstance(value, dict):
        value = {}
    result = dict(value)
    jq_security = _from_qmt_security(security or qmt_security)
    display_name = result.get("display_name") or result.get("InstrumentName") or jq_security
    name = result.get("name") or result.get("InstrumentID") or jq_security.split(".", 1)[0]
    start_date = result.get("start_date")
    if start_date in (None, ""):
        start_date = _security_date_value(result.get("OpenDate"))
    end_date = result.get("end_date")
    if end_date in (None, ""):
        end_date = _security_date_value(result.get("ExpireDate"), expire=True)
    if end_date in (None, ""):
        end_date = "2200-01-01T00:00:00"
    result["display_name"] = _basic_value(display_name)
    result["name"] = _basic_value(name)
    result["start_date"] = _basic_value(start_date)
    result["end_date"] = _basic_value(end_date)
    result["type"] = _normalize_security_type(context_info, qmt_security, result)
    result.setdefault("subtype", None)
    result.setdefault("parent", None)
    result["code"] = jq_security
    result["qmt_code"] = qmt_security
    result["security"] = jq_security
    result["qmt_security"] = qmt_security
    return result


def _query_security_info(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    security = payload.get("security") or payload.get("stockcode")
    if not security:
        return _error("BAD_REQUEST", "security_info missing security", payload.get("request_id"))
    qmt_security = _to_qmt_security(security)
    try:
        if context_info is None:
            return _context_not_ready(payload)
        getter = getattr(context_info, "get_instrument_detail", None)
        if getter is None:
            getter = getattr(context_info, "get_instrumentdetail", None)
        if getter is None:
            return _error("NOT_IMPLEMENTED", "missing get_instrument_detail", payload.get("request_id"))
        try:
            info = getter(qmt_security, True)
        except TypeError:
            info = getter(qmt_security)
        value = _normalize_security_info_value(context_info, security, qmt_security, info)
        return _ok(value, payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_instrument_detail unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_instrument_detail failed: %s", exc)
        return _error("SECURITY_INFO_FAILED", str(exc), payload.get("request_id"))


def _ensure_cache(payload: Dict[str, Any]) -> Dict[str, Any]:
    security = payload.get("security") or payload.get("stockcode")
    if not security:
        return _error("BAD_REQUEST", "ensure_cache missing security", payload.get("request_id"))
    qmt_security = _to_qmt_security(security)
    period = _qmt_period(payload.get("frequency") or payload.get("period") or "1m")
    start = payload.get("start") or payload.get("start_date") or payload.get("start_time") or ""
    end = payload.get("end") or payload.get("end_date") or payload.get("end_time") or ""
    try:
        return _ok(_call_download_history_data(qmt_security, period, start, end, payload), payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("download_history_data unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("download_history_data failed: %s", exc)
        return _error("ENSURE_CACHE_FAILED", str(exc), payload.get("request_id"))


def _sector_for_types(types: Any) -> str:
    text = str(types or "stock").lower()
    if "300" in text:
        return "\u6caa\u6df1300"
    if "500" in text:
        return "\u4e2d\u8bc1500"
    if "1000" in text:
        return "\u4e2d\u8bc11000"
    if "50" in text:
        return "\u4e0a\u8bc150"
    if "etf" in text:
        return "\u6caa\u6df1ETF"
    if "fund" in text:
        return "\u6caa\u6df1\u57fa\u91d1"
    return "\u6caa\u6df1A\u80a1"


def _security_result_type(types: Any) -> str:
    text = str(types or "stock").lower()
    if "etf" in text:
        return "etf"
    if "fund" in text:
        return "fund"
    if "index" in text:
        return "index"
    return "stock"


def _security_date_value(value: Any, *, expire: bool = False) -> Any:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if expire and text in ("99999999", "99991231"):
        return None
    if text in ("0", "0.0"):
        return "1970-01-01T00:00:00"
    digits = _date_digits(text)
    if len(digits) >= 8:
        date = digits[:8]
        return "%s-%s-%sT00:00:00" % (date[:4], date[4:6], date[6:8])
    return _basic_value(value)


def _security_listing_payload(context_info: Any, values: List[Any], types: Any) -> Dict[str, Any]:
    result_type = _security_result_type(types)
    rows = []
    getter = getattr(context_info, "get_instrument_detail", None)
    if getter is None:
        getter = getattr(context_info, "get_instrumentdetail", None)
    for item in values:
        qmt_security = str(item)
        jq_security = _from_qmt_security(qmt_security)
        info = {}
        if getter is not None:
            try:
                info = getter(qmt_security, True)
            except TypeError:
                try:
                    info = getter(qmt_security)
                except Exception:
                    info = {}
            except Exception:
                info = {}
        if not isinstance(info, dict):
            info = {}
        display_name = info.get("InstrumentName") or qmt_security
        name = info.get("InstrumentID") or qmt_security.split(".", 1)[0]
        rows.append(
            [
                jq_security,
                _basic_value(display_name),
                _basic_value(name),
                _security_date_value(info.get("OpenDate")),
                _security_date_value(info.get("ExpireDate"), expire=True),
                result_type,
                qmt_security,
            ]
        )
    return {
        "dtype": "dataframe",
        "columns": ["display_name", "name", "start_date", "end_date", "type", "qmt_code"],
        "records": rows,
    }


def _query_all_securities(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    sector = payload.get("sector") or payload.get("sector_name") or _sector_for_types(payload.get("types"))
    try:
        if context_info is None:
            return _context_not_ready(payload)
        values = context_info.get_stock_list_in_sector(str(sector)) or []
        return _ok(_security_listing_payload(context_info, values, payload.get("types")), payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_stock_list_in_sector unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_stock_list_in_sector failed: %s", exc)
        return _error("ALL_SECURITIES_FAILED", str(exc), payload.get("request_id"))


def _call_optional_callable(owner: Any, name: str) -> Optional[Any]:
    if owner is None:
        return None
    value = getattr(owner, name, None)
    if callable(value):
        return value
    return None


def _index_weight_codes_from_data(data: Any) -> List[str]:
    if data is None:
        return []
    if isinstance(data, dict):
        for key in ("stocks", "values", "codes", "stock_codes"):
            value = data.get(key)
            if isinstance(value, (list, tuple, set)):
                return [_from_qmt_security(_to_qmt_security(item)) for item in value if item]
        return [_from_qmt_security(_to_qmt_security(item)) for item in data.keys() if item]
    if isinstance(data, (list, tuple, set)):
        result = []
        for item in data:
            if isinstance(item, dict):
                code = (
                    item.get("code")
                    or item.get("stock_code")
                    or item.get("security")
                    or item.get("instrument")
                    or item.get("InstrumentID")
                )
            else:
                code = item
            if code:
                result.append(_from_qmt_security(_to_qmt_security(code)))
        return result
    columns = getattr(data, "columns", None)
    if columns is not None:
        try:
            column_names = [str(item) for item in list(columns)]
        except Exception:
            column_names = []
        for key in ("code", "stock_code", "security", "instrument", "InstrumentID"):
            if key in column_names:
                try:
                    values = list(data[key])
                except Exception:
                    values = []
                if values:
                    return [_from_qmt_security(_to_qmt_security(item)) for item in values if item]
    index = getattr(data, "index", None)
    if index is not None:
        try:
            values = list(index.tolist())
        except Exception:
            try:
                values = list(index)
            except Exception:
                values = []
        if values:
            return [_from_qmt_security(_to_qmt_security(item)) for item in values if item]
    return []


def _query_index_weight_stocks(context_info: Any, index_symbol: Any) -> List[str]:
    qmt_index = _to_qmt_security(index_symbol)
    if INDEX_WEIGHT_DOWNLOAD_BEFORE_READ:
        downloader = globals().get("download_index_weight") or getattr(builtins, "download_index_weight", None)
        if callable(downloader):
            try:
                downloader()
            except TypeError:
                try:
                    downloader(qmt_index)
                except Exception:
                    LOGGER.exception("download_index_weight failed")
            except Exception:
                LOGGER.exception("download_index_weight failed")
    readers = []
    for owner in (context_info, builtins):
        reader = _call_optional_callable(owner, "get_index_weight")
        if reader is not None:
            readers.append(reader)
    reader = globals().get("get_index_weight")
    if callable(reader):
        readers.append(reader)
    for reader in readers:
        try:
            data = reader(qmt_index)
        except TypeError:
            try:
                data = reader(str(qmt_index))
            except Exception:
                LOGGER.exception("get_index_weight failed")
                continue
        except Exception:
            LOGGER.exception("get_index_weight failed")
            continue
        codes = _index_weight_codes_from_data(data)
        if codes:
            return codes
    return []


def _sector_for_index(index_symbol: Any) -> str:
    text = str(index_symbol or "")
    mapping = {
        "000016": "\u4e0a\u8bc150",
        "000300": "\u6caa\u6df1300",
        "399300": "\u6caa\u6df1300",
        "000905": "\u4e2d\u8bc1500",
        "399905": "\u4e2d\u8bc1500",
        "000852": "\u4e2d\u8bc11000",
    }
    for key, value in mapping.items():
        if text.startswith(key):
            return value
    return text or "\u6caa\u6df1300"


def _query_index_stocks(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    index_symbol = payload.get("index_symbol")
    if index_symbol:
        values = _query_index_weight_stocks(context_info, index_symbol)
        if values:
            return _ok(
                {
                    "stocks": values,
                    "values": values,
                    "qmt_stocks": [_to_qmt_security(item) for item in values],
                    "source": "get_index_weight",
                },
                payload.get("request_id"),
            )
    sector = payload.get("sector") or payload.get("sector_name") or _sector_for_index(index_symbol)
    try:
        if context_info is None:
            return _context_not_ready(payload)
        values = context_info.get_stock_list_in_sector(str(sector)) or []
        stocks = [_from_qmt_security(item) for item in values]
        return _ok(
            {
                "stocks": stocks,
                "values": stocks,
                "qmt_stocks": list(values),
                "sector": sector,
                "source": "sector_fallback",
            },
            payload.get("request_id"),
        )
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_stock_list_in_sector index unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_stock_list_in_sector index failed: %s", exc)
        return _error("INDEX_STOCKS_FAILED", str(exc), payload.get("request_id"))


def _query_split_dividend(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    security = payload.get("security") or payload.get("stockcode")
    if not security:
        return _error("BAD_REQUEST", "get_split_dividend missing security", payload.get("request_id"))
    qmt_security = _to_qmt_security(security)
    start = payload.get("start") or payload.get("start_date") or ""
    end = payload.get("end") or payload.get("end_date") or ""
    try:
        if context_info is None:
            return _context_not_ready(payload)
        raw = context_info.get_divid_factors(qmt_security) or {}
        events = []
        for key in sorted(raw.keys()):
            if not _date_in_range(key, start, end):
                continue
            item = raw.get(key)
            events.append(_split_dividend_row(qmt_security, security, key, item))
        return _ok({"events": events}, payload.get("request_id"))
    except QmtApiUnavailable as exc:
        LOGGER.exception("get_divid_factors unavailable: %s", exc)
        return _error("QMT_API_NOT_READY", str(exc), payload.get("request_id"))
    except Exception as exc:
        LOGGER.exception("get_divid_factors failed: %s", exc)
        return _error("SPLIT_DIVIDEND_FAILED", str(exc), payload.get("request_id"))


def _call_passorder(
    passer: Any,
    op_type: int,
    account_id: str,
    qmt_security: str,
    pr_type: int,
    price: float,
    amount: int,
    strategy_name: str,
    qmt_user_order_id: Optional[str],
    context_info: Any,
    quick_trade: int,
) -> Dict[str, Any]:
    fallback_error = None
    if PASSORDER_USE_REMARK_SIGNATURE and qmt_user_order_id:
        try:
            order_ref = passer(
                op_type,
                1101,
                account_id,
                qmt_security,
                pr_type,
                float(price),
                amount,
                strategy_name,
                quick_trade,
                qmt_user_order_id,
                context_info,
            )
            return {
                "order_ref": order_ref,
                "passorder_signature": "official_user_order_id",
                "qmt_user_order_id": qmt_user_order_id,
                "fallback_error": None,
            }
        except TypeError as exc:
            fallback_error = str(exc)
            _emit("warning", "passorder with remark signature failed; fallback to article signature error=%s", exc)
    remark_or_strategy = strategy_name or "qmt"
    try:
        order_ref = passer(
            op_type,
            1101,
            account_id,
            qmt_security,
            pr_type,
            float(price),
            amount,
            remark_or_strategy,
            quick_trade,
            context_info,
        )
        return {
            "order_ref": order_ref,
            "passorder_signature": "article",
            "qmt_user_order_id": qmt_user_order_id or "",
            "fallback_error": fallback_error,
        }
    except TypeError:
        if remark_or_strategy != strategy_name:
            order_ref = passer(
                op_type,
                1101,
                account_id,
                qmt_security,
                pr_type,
                float(price),
                amount,
                strategy_name,
                quick_trade,
                context_info,
            )
            return {
                "order_ref": order_ref,
                "passorder_signature": "article_strategy_name",
                "qmt_user_order_id": qmt_user_order_id or "",
                "fallback_error": fallback_error,
            }
        raise


def _request_quick_trade(payload: Dict[str, Any]) -> int:
    value = None
    if ALLOW_PASSORDER_QUICK_TRADE_OVERRIDE:
        value = payload.get("quick_trade")
        if value is None:
            value = payload.get("quickTrade")
    if value is None:
        value = PASSORDER_QUICK_TRADE
    return int(value)


def _ensure_context_account(context_info: Any, account_id: str) -> None:
    if context_info is None or _is_placeholder_account_id(account_id):
        return
    try:
        setter = getattr(context_info, "set_account", None)
        if setter is not None:
            setter(account_id)
            _emit("info", "ContextInfo.set_account before passorder success account=%s", account_id)
    except Exception as exc:
        _emit("warning", "ContextInfo.set_account before passorder failed account=%s error=%s", account_id, exc)
    try:
        context_info.accountID = account_id
        _emit("info", "ContextInfo.accountID before passorder assigned account=%s", account_id)
    except Exception as exc:
        _emit("warning", "ContextInfo.accountID before passorder assign failed account=%s error=%s", account_id, exc)


def _place_order(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id")
    if not ENABLE_TRADING:
        return _error("TRADING_DISABLED", "trading is disabled", request_id)
    if context_info is None:
        return _error("QMT_NOT_READY", "ContextInfo is required before trading", request_id)
    account = _resolve_account(payload)
    if _is_placeholder_account_id(account["account_id"]):
        return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id before trading", request_id)
    security = str(payload.get("security") or payload.get("stock") or payload.get("stockcode") or "")
    qmt_security = _to_qmt_security(security)
    side = str(payload.get("side") or "BUY").upper()
    amount = int(payload.get("amount") or payload.get("volume") or 0)
    price = payload.get("price")
    style = payload.get("style") or {}
    if price is None:
        price = style.get("price") or style.get("protect_price")
    if not security or amount <= 0 or price is None:
        return _error("BAD_REQUEST", "place_order requires security, amount and price", request_id)
    pr_type = int(payload.get("pr_type") or payload.get("prType") or style.get("pr_type") or style.get("prType") or 11)
    op_type = 23 if side == "BUY" else 24
    strategy_name = str(payload.get("strategy_name") or payload.get("strategy") or "qmt")
    order_remark = _compose_order_remark(payload)
    qmt_user_order_id = _compose_qmt_user_order_id(payload, order_remark)
    virtual_account_id = _resolve_virtual_account_id(payload)
    quick_trade = _request_quick_trade(payload)
    _ensure_context_account(context_info, account["account_id"])
    _emit(
        "info",
        "passorder begin account=%s account_type=%s op_type=%s security=%s pr_type=%s price=%s amount=%s "
        "strategy=%s quick_trade=%s remark=%s qmt_user_order_id=%s",
        account["account_id"],
        account["account_type"],
        op_type,
        qmt_security,
        pr_type,
        float(price),
        amount,
        strategy_name,
        quick_trade,
        order_remark or "",
        qmt_user_order_id,
    )
    try:
        passer = _qmt_global("passorder")
        passorder_result = _call_passorder(
            passer,
            op_type,
            account["account_id"],
            qmt_security,
            pr_type,
            float(price),
            amount,
            strategy_name,
            qmt_user_order_id,
            context_info,
            quick_trade,
        )
        order_ref = passorder_result.get("order_ref")
    except Exception as exc:
        LOGGER.exception("passorder failed: %s", exc)
        return _error("PLACE_ORDER_FAILED", str(exc), request_id)
    _emit(
        "warning" if order_ref in (None, "", 0, "0") else "info",
        "passorder returned account=%s security=%s op_type=%s pr_type=%s price=%s amount=%s "
        "return=%s return_type=%s signature=%s qmt_user_order_id=%s",
        account["account_id"],
        qmt_security,
        op_type,
        pr_type,
        float(price),
        amount,
        _basic_value(order_ref),
        type(order_ref).__name__,
        passorder_result.get("passorder_signature"),
        passorder_result.get("qmt_user_order_id"),
    )
    order_tag = {
        "security": _from_qmt_security(security),
        "qmt_security": qmt_security,
        "side": side,
        "amount": amount,
        "price": float(price),
        "pr_type": pr_type,
        "quick_trade": quick_trade,
        "qmt_user_order_id": qmt_user_order_id,
        "order_remark": order_remark,
        "remark": order_remark,
        "strategy_name": strategy_name,
        "sub_account_id": virtual_account_id,
        "virtual_account_id": virtual_account_id,
        "request_id": request_id,
        "created_at": time.time(),
    }
    if order_ref in (None, "", 0, "0"):
        # Big QMT commonly returns 0 even when the order is accepted. Keep a
        # pending tag so the later /orders poll can attach the full BulletTrade
        # remark and virtual account to the newly visible order id.
        _remember_order_tag(None, order_tag)
        empty_return_value = {
            "order_id": "",
            "order_ref": "",
            "passorder_return": _basic_value(order_ref),
            "passorder_return_type": type(order_ref).__name__,
            "passorder_return_is_none": order_ref is None,
            "passorder_signature": passorder_result.get("passorder_signature"),
            "passorder_fallback_error": passorder_result.get("fallback_error"),
            "security": _from_qmt_security(security),
            "qmt_security": qmt_security,
            "account_id": account["account_id"],
            "account_type": account["account_type"],
            "side": side,
            "amount": amount,
            "price": float(price),
            "pr_type": pr_type,
            "quick_trade": quick_trade,
            "strategy_name": strategy_name,
            "qmt_user_order_id": qmt_user_order_id,
            "order_remark": order_remark,
            "remark": order_remark,
            "sub_account_id": virtual_account_id,
            "virtual_account_id": virtual_account_id,
            "status": "submit_unknown",
            "submit_unknown": True,
            "order_tag_recorded": True,
            "warning": (
                "passorder returned empty/zero order reference; treating as submit_unknown "
                "to match the article sample and allow order polling"
            ),
        }
        if PASSORDER_EMPTY_RETURN_AS_SUBMIT_UNKNOWN:
            return _ok(empty_return_value, request_id)
        response = _error(
            "PLACE_ORDER_REJECTED",
            "passorder returned empty/zero order reference; check QMT trading permission, account mode and PASSORDER_QUICK_TRADE",
            request_id,
        )
        response["details"] = empty_return_value
        return response
    _remember_order_tag(order_ref if order_ref else None, order_tag)
    result = {
        "order_id": str(order_ref) if order_ref else "",
        "order_ref": str(order_ref) if order_ref else "",
        "passorder_return": _basic_value(order_ref),
        "passorder_return_type": type(order_ref).__name__,
        "passorder_return_is_none": order_ref is None,
        "passorder_signature": passorder_result.get("passorder_signature"),
        "passorder_fallback_error": passorder_result.get("fallback_error"),
        "security": _from_qmt_security(security),
        "qmt_security": qmt_security,
        "account_id": account["account_id"],
        "account_type": account["account_type"],
        "side": side,
        "amount": amount,
        "price": float(price),
        "pr_type": pr_type,
        "quick_trade": quick_trade,
        "qmt_user_order_id": qmt_user_order_id,
        "order_remark": order_remark,
        "remark": order_remark,
        "strategy_name": strategy_name,
        "sub_account_id": virtual_account_id,
        "virtual_account_id": virtual_account_id,
        "order_tag_recorded": True,
        "order_tag_store": _order_tag_store_path(),
    }
    return _ok(result, request_id)


def _cancel_order(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id")
    if not ENABLE_CANCEL_ORDER:
        return _error("CANCEL_ORDER_DISABLED", "cancel_order is disabled", request_id)
    if context_info is None:
        return _error("QMT_NOT_READY", "ContextInfo is required before cancel_order", request_id)
    account = _resolve_account(payload)
    if _is_placeholder_account_id(account["account_id"]):
        return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id before cancel_order", request_id)
    order_id = str(payload.get("order_id") or payload.get("order_sys_id") or "")
    if not order_id:
        return _error("MISSING_ORDER_ID", "cancel_order requires order_id/order_sys_id", request_id)
    try:
        sub_account_id = _resolve_virtual_account_id(payload)
        if sub_account_id:
            matched_orders = _query_orders(account["account_id"], account["account_type"], {"order_id": order_id})
            if matched_orders and not _filter_virtual_account_rows(matched_orders, payload):
                return _error(
                    "VIRTUAL_ACCOUNT_MISMATCH",
                    "order %s does not belong to sub_account_id=%s" % (order_id, sub_account_id),
                    request_id,
                )
        can_cancel = _qmt_global("can_cancel_order")
        cancel_fn = _qmt_global("cancel")
        if not can_cancel(order_id, account["account_id"], account["account_type"]):
            return _error("ORDER_NOT_CANCELABLE", "order is not cancelable", request_id)
        accepted = cancel_fn(order_id, account["account_id"], account["account_type"], context_info)
    except Exception as exc:
        LOGGER.exception("cancel failed: %s", exc)
        return _error("CANCEL_ORDER_FAILED", str(exc), request_id)
    return _ok(
        {
            "order_id": order_id,
            "success": bool(accepted is not False),
            "account_id": account["account_id"],
            "account_type": account["account_type"],
        },
        request_id,
    )


def _cancel_by_rule(context_info: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id")
    if not ENABLE_RULE_CANCEL:
        return _error("DANGEROUS_OPERATION_DISABLED", "rule_cancel is disabled", request_id)
    if not ENABLE_CANCEL_ORDER:
        return _error("CANCEL_ORDER_DISABLED", "cancel_order is disabled", request_id)
    if context_info is None:
        return _error("QMT_NOT_READY", "ContextInfo is required before cancel_order", request_id)
    account = _resolve_account(payload)
    if _is_placeholder_account_id(account["account_id"]):
        return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id before cancel_order", request_id)
    security = str(payload.get("security") or payload.get("stock") or payload.get("stockcode") or "")
    amount = int(payload.get("amount") or payload.get("volume") or 0)
    if not security:
        return _error("BAD_REQUEST", "rule_cancel requires security/stock", request_id)
    target_security = _from_qmt_security(_to_qmt_security(security))
    try:
        can_cancel = _qmt_global("can_cancel_order")
        cancel_fn = _qmt_global("cancel")
        canceled = []
        inspected = []
        for order in _query_orders(account["account_id"], account["account_type"], payload):
            order_id = str(order.get("order_id") or "")
            order_security = str(order.get("security") or "")
            order_total = int(order.get("volume_left") or 0) + int(order.get("filled") or 0)
            inspected.append(
                {
                    "order_id": order_id,
                    "security": order_security,
                    "volume_left": order.get("volume_left"),
                    "filled": order.get("filled"),
                    "raw_status": order.get("raw_status"),
                    "order_remark": order.get("order_remark"),
                    "sub_account_id": order.get("sub_account_id"),
                }
            )
            if not order_id or order_security != target_security:
                continue
            if amount > 0 and order_total != amount:
                continue
            if not can_cancel(order_id, account["account_id"], account["account_type"]):
                continue
            accepted = cancel_fn(order_id, account["account_id"], account["account_type"], context_info)
            canceled.append(
                {
                    "order_id": order_id,
                    "security": order_security,
                    "success": bool(accepted is not False),
                }
            )
        if not canceled:
            return _error(
                "ORDER_NOT_FOUND",
                "no cancelable order matched security=%s amount=%s" % (target_security, amount),
                request_id,
            )
        return _ok({"canceled_orders": canceled, "inspected_orders": inspected}, request_id)
    except Exception as exc:
        LOGGER.exception("rule cancel failed: %s", exc)
        return _error("CANCEL_ORDER_FAILED", str(exc), request_id)


def _csv_list(value: Any, default: List[Any], none_names: List[str]) -> List[Any]:
    if value is None or value == "":
        return list(default)
    if isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        raw_items = str(value).split(",")
    items = []
    for raw in raw_items:
        text = str(raw).strip()
        if text.lower() in none_names:
            items.append(None)
        elif text:
            items.append(text)
    return items or list(default)


def _query_trade_detail_debug(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id")
    if not ENABLE_TRADE_DETAIL_DEBUG:
        return _error("DEBUG_DISABLED", "trade_detail debug is disabled", request_id)
    account = _resolve_account(payload)
    if _is_placeholder_account_id(account["account_id"]):
        return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for debug", request_id)
    detail_types = _csv_list(
        payload.get("detail_types") or payload.get("types") or payload.get("detail_type"),
        ["account", "position", "order", "deal", "trade"],
        [],
    )
    sources = _csv_list(
        payload.get("sources") or payload.get("source"),
        ["qmt", None],
        ["default", "none", "null", "no_source"],
    )
    limit = max(0, min(_to_int(payload.get("limit"), TRADE_DETAIL_DEBUG_MAX_ROWS), 20))
    results = []
    for detail_type in detail_types:
        for source in sources:
            started = time.time()
            result = _try_trade_detail(account["account_id"], account["account_type"], str(detail_type), source)
            elapsed_ms = round((time.time() - started) * 1000, 3)
            rows = result["rows"]
            samples = [_basic_value(row) for row in rows[:limit]]
            results.append(
                {
                    "detail_type": str(detail_type),
                    "source": source if source is not None else "default",
                    "ok": result["ok"],
                    "error": result["error"],
                    "row_count": len(rows),
                    "elapsed_ms": elapsed_ms,
                    "samples": samples,
                }
            )
    return _ok(
        {
            "account_id": account["account_id"],
            "account_type": account["account_type"],
            "detail_types": [str(item) for item in detail_types],
            "sources": [item if item is not None else "default" for item in sources],
            "limit": limit,
            "results": results,
        },
        request_id,
    )


def _dispatch_qmt_action(context_info: Any, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = payload.get("request_id")
    if action in ("snapshot", "current_tick", "live_current"):
        value = _get_full_tick(context_info, payload)
        if value.get("ok") is False and value.get("code"):
            return value
        return _ok(value, request_id)
    if action == "account":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for account query", request_id)
        return _ok(_query_account(account["account_id"], account["account_type"]), request_id)
    if action == "money_total":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for money query", request_id)
        account_info = _query_account(account["account_id"], account["account_type"])
        return _ok({"total_value": account_info.get("total_value"), "account": account_info}, request_id)
    if action == "money_available":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for money query", request_id)
        account_info = _query_account(account["account_id"], account["account_type"])
        return _ok({"available_cash": account_info.get("available_cash"), "account": account_info}, request_id)
    if action == "positions":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for positions query", request_id)
        return _ok({"positions": _query_positions(account["account_id"], account["account_type"])}, request_id)
    if action == "orders":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for orders query", request_id)
        orders = _query_orders(account["account_id"], account["account_type"], payload)
        return _ok({"orders": orders}, request_id)
    if action == "trades":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for trades query", request_id)
        return _ok({"trades": _query_trades(account["account_id"], account["account_type"], payload)}, request_id)
    if action == "order_status":
        account = _resolve_account(payload)
        if _is_placeholder_account_id(account["account_id"]):
            return _error("ACCOUNT_NOT_CONFIGURED", "set ACCOUNT_ID or pass account_id for order_status query", request_id)
        order_id = str(payload.get("order_id") or payload.get("order_sys_id") or "")
        for order in _query_orders(account["account_id"], account["account_type"], payload):
            if str(order.get("order_id")) == order_id:
                return _ok({"order": order}, request_id)
        return _error("ORDER_NOT_FOUND", "order not found: %s" % order_id, request_id)
    if action == "place_order":
        return _place_order(context_info, payload)
    if action == "cancel_order":
        return _cancel_order(context_info, payload)
    if action == "rule_cancel":
        return _cancel_by_rule(context_info, payload)
    if action == "debug_trade_detail":
        return _query_trade_detail_debug(payload)
    if action == "history":
        return _query_history(context_info, payload)
    if action == "trade_days":
        return _query_trade_days(context_info, payload)
    if action == "security_info":
        return _query_security_info(context_info, payload)
    if action == "ensure_cache":
        return _ensure_cache(payload)
    if action == "all_securities":
        return _query_all_securities(context_info, payload)
    if action == "index_stocks":
        return _query_index_stocks(context_info, payload)
    if action == "split_dividend":
        return _query_split_dividend(context_info, payload)
    if action == "cancel_all" and not ENABLE_CANCEL_ALL:
        return _error("DANGEROUS_OPERATION_DISABLED", "cancel_all is disabled by default", request_id)
    return _error("NOT_FOUND", "unknown action: %s" % action, request_id)


ACCOUNT_QUERY_ACTIONS = set(
    [
        "account",
        "money_total",
        "money_available",
        "positions",
        "orders",
        "trades",
        "order_status",
        "debug_trade_detail",
    ]
)


class _GatewayRuntime:
    def __init__(self) -> None:
        self.context_info = None
        self.init_called = False
        self.http_started_from = None
        self.started_at = time.time()
        self.last_error = None
        self.last_success_at = None
        self.request_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.http_server = None
        self.ioloop = None
        self.http_thread = None
        self.direct_dispatch = False

    def _dispatch_now(self, action: str, payload: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        try:
            _emit(
                "info",
                "dispatch direct action=%s request_id=%s context_ready=%s",
                action,
                request_id,
                self.context_info is not None,
            )
            response = _dispatch_qmt_action(self.context_info, action, payload)
            self.last_success_at = time.time()
            _emit(
                "info",
                "dispatch done action=%s request_id=%s ok=%s code=%s",
                action,
                request_id,
                response.get("ok"),
                response.get("code") or "",
            )
            return response
        except Exception as exc:
            LOGGER.exception("direct dispatch failed: %s", exc)
            _emit("error", "dispatch failed action=%s request_id=%s error=%s", action, request_id, exc)
            self.last_error = str(exc)
            if isinstance(exc, QmtApiUnavailable):
                return _error("QMT_API_NOT_READY", str(exc), request_id)
            return _error(
                "QMT_ACTION_FAILED",
                "%s\n%s" % (exc, traceback.format_exc()),
                request_id,
            )

    def submit(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = payload.get("request_id") or str(uuid4())
        payload["request_id"] = request_id
        if self.direct_dispatch:
            return self._dispatch_now(action, payload, request_id)
        if DIRECT_DISPATCH_ACCOUNT_APIS and action in ACCOUNT_QUERY_ACTIONS:
            _emit(
                "info",
                "bypass queue for account api action=%s request_id=%s",
                action,
                request_id,
            )
            return self._dispatch_now(action, payload, request_id)
        if self.context_info is None:
            return self._dispatch_now(action, payload, request_id)
        job = {
            "action": action,
            "payload": payload,
            "event": threading.Event(),
            "response": None,
        }
        try:
            self.request_queue.put_nowait(job)
        except queue.Full:
            _emit("warning", "reject action=%s request_id=%s reason=queue_full", action, request_id)
            return _error("GATEWAY_BUSY", "request queue is full", request_id)
        _emit("info", "enqueue action=%s request_id=%s queue_size=%s", action, request_id, self.request_queue.qsize())
        if not job["event"].wait(REQUEST_TIMEOUT_SECONDS):
            _emit("warning", "timeout action=%s request_id=%s", action, request_id)
            return _error("GATEWAY_TIMEOUT", "timed out waiting for QMT thread", request_id)
        return job["response"] or _error("EMPTY_RESPONSE", "QMT thread returned no response", request_id)

    def drain(self, context_info: Any, limit: int = 50) -> None:
        self.context_info = context_info
        for _ in range(limit):
            try:
                job = self.request_queue.get_nowait()
            except queue.Empty:
                return
            try:
                response = _dispatch_qmt_action(context_info, job["action"], job["payload"])
                self.last_success_at = time.time()
            except QmtApiUnavailable as exc:
                LOGGER.exception("dispatch failed: %s", exc)
                self.last_error = str(exc)
                response = _error("QMT_API_NOT_READY", str(exc), job["payload"].get("request_id"))
            except Exception as exc:
                LOGGER.exception("dispatch failed: %s", exc)
                self.last_error = str(exc)
                response = _error(
                    "QMT_ACTION_FAILED",
                    "%s\n%s" % (exc, traceback.format_exc()),
                    job["payload"].get("request_id"),
                )
            job["response"] = response
            job["event"].set()

    def health(self) -> Dict[str, Any]:
        context_ready = self.context_info is not None
        qmt_api_ready = _qmt_global_available("get_trade_detail_data")
        account_configured = not _is_placeholder_account_id(ACCOUNT_ID)
        return {
            "ready": context_ready or qmt_api_ready,
            "http_alive": self.ioloop is not None,
            "init_called": self.init_called,
            "context_ready": context_ready,
            "qmt_api_ready": qmt_api_ready,
            "data_context_ready": context_ready,
            "trading_context_ready": context_ready,
            "qmt_apis": {
                "get_trade_detail_data": _qmt_global_available("get_trade_detail_data"),
                "download_history_data": _qmt_global_available("download_history_data"),
                "passorder": _qmt_global_available("passorder"),
                "cancel": _qmt_global_available("cancel"),
                "can_cancel_order": _qmt_global_available("can_cancel_order"),
            },
            "backend_type": "big_qmt",
            "strategy": "bt_big_qmt_gateway",
            "gateway_build_id": GATEWAY_BUILD_ID,
            "listen": "%s:%s" % (LISTEN_HOST, LISTEN_PORT),
            "log_file": LOG_FILE,
            "account_id": ACCOUNT_ID,
            "account_type": ACCOUNT_TYPE,
            "account_configured": account_configured,
            "trading_enabled": ENABLE_TRADING,
            "cancel_order_enabled": ENABLE_CANCEL_ORDER,
            "cancel_all_enabled": ENABLE_CANCEL_ALL,
            "rule_cancel_enabled": ENABLE_RULE_CANCEL,
            "trade_detail_debug_enabled": ENABLE_TRADE_DETAIL_DEBUG,
            "run_http_in_background_thread": RUN_HTTP_IN_BACKGROUND_THREAD,
            "direct_dispatch": self.direct_dispatch,
            "direct_dispatch_account_apis": DIRECT_DISPATCH_ACCOUNT_APIS,
            "http_started_from": self.http_started_from,
            "module_autostart_enabled": AUTO_START_HTTP_ON_MODULE_LOAD,
            "stop_http_on_qmt_stop": STOP_HTTP_ON_QMT_STOP,
            "queue_size": self.request_queue.qsize(),
            "queue_max_size": MAX_QUEUE_SIZE,
            "last_error": self.last_error,
            "last_success_at": self.last_success_at,
            "uptime_seconds": max(0.0, time.time() - self.started_at),
        }


RUNTIME = _GatewayRuntime()


class _GatewayHandler(tornado.web.RequestHandler):
    def set_default_headers(self) -> None:
        self.set_header("Content-Type", "application/json; charset=utf-8")

    def get(self) -> None:
        route = self.request.path
        query = self._query_payload()
        request_id = self._request_id()
        _emit("info", "http GET route=%s request_id=%s remote=%s", route, request_id, self.request.remote_ip)
        if route == "/health":
            self._send_json(200, _ok(RUNTIME.health(), request_id))
            return
        action = _route_to_action(route, query)
        if action:
            query.setdefault("request_id", request_id)
            self._handle_action(action, query)
            return
        self._send_json(404, _error("NOT_FOUND", "unknown route: %s" % route, request_id))

    def post(self) -> None:
        route = self.request.path
        query = self._query_payload()
        payload = self._read_json()
        payload.update(query)
        request_id = self._request_id()
        payload.setdefault("request_id", request_id)
        _emit("info", "http POST route=%s request_id=%s remote=%s", route, request_id, self.request.remote_ip)
        action = _route_to_action(route, payload)
        if not action:
            self._send_json(404, _error("NOT_FOUND", "unknown route: %s" % route, request_id))
            return
        self._handle_action(action, payload)

    def _handle_action(self, action: str, payload: Dict[str, Any]) -> None:
        request_id = payload.get("request_id") or self._request_id()
        if not _check_password(self.request.headers):
            _emit("warning", "auth failed action=%s request_id=%s remote=%s", action, request_id, self.request.remote_ip)
            self._send_json(401, _error("AUTH_FAILED", "gateway password mismatch", request_id))
            return
        payload.setdefault("request_id", request_id)
        response = RUNTIME.submit(action, payload)
        status = 200 if response.get("ok") else 400
        if response.get("code") in ("NOT_FOUND", "NOT_IMPLEMENTED"):
            status = 404
        _emit(
            "info",
            "http response action=%s request_id=%s status=%s ok=%s code=%s",
            action,
            request_id,
            status,
            response.get("ok"),
            response.get("code") or "",
        )
        self._send_json(status, response)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        request_id = payload.get("request_id") if isinstance(payload, dict) else ""
        body = json.dumps(payload, ensure_ascii=False)
        body_bytes = body.encode("utf-8")
        self.set_status(status)
        self.set_header("Content-Length", str(len(body_bytes)))
        self.set_header("Connection", "close")
        _emit(
            "info",
            "http send begin route=%s request_id=%s status=%s bytes=%s",
            self.request.path,
            request_id or "",
            status,
            len(body_bytes),
        )
        self.write(body)
        self.finish()
        _emit(
            "info",
            "http send done route=%s request_id=%s status=%s",
            self.request.path,
            request_id or "",
            status,
        )

    def _read_json(self) -> Dict[str, Any]:
        length = len(self.request.body or b"")
        if length > MAX_REQUEST_BODY_BYTES:
            return {}
        if length <= 0:
            return {}
        try:
            value = json.loads(self.request.body.decode("utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _query_payload(self) -> Dict[str, Any]:
        payload = {}
        for key in self.request.query_arguments:
            values = self.get_query_arguments(key)
            if values:
                payload[key] = values[-1]
        return payload

    def _request_id(self) -> str:
        return self.request.headers.get("X-BulletTrade-Request-Id") or str(uuid4())


def _route_to_action(route: str, payload: Dict[str, Any]) -> Optional[str]:
    mapping = {
        "/data/history": "history",
        "/data/snapshot": "snapshot",
        "/data/current_tick": "current_tick",
        "/data/live_current": "live_current",
        "/data/trade_days": "trade_days",
        "/data/security_info": "security_info",
        "/data/ensure_cache": "ensure_cache",
        "/data/all_securities": "all_securities",
        "/data/index_stocks": "index_stocks",
        "/data/split_dividend": "split_dividend",
        "/account": "account",
        "/positions": "positions",
        "/orders": "orders",
        "/trades": "trades",
        "/order_status": "order_status",
        "/place_order": "place_order",
        "/cancel_order": "cancel_order",
        "/debug/trade_detail": "debug_trade_detail",
        "/debug/qmt_trade_detail": "debug_trade_detail",
        "/api/holding": "positions",
        "/api/money/total": "money_total",
        "/api/money/available": "money_available",
        "/api/market/full_tick": "snapshot",
        "/api/order/status": "orders",
        "/api/order/cancel_all": "cancel_all",
        "/api/order/cancel_order": "rule_cancel",
    }
    if route == "/api/order/buy":
        payload["side"] = "BUY"
        return "place_order"
    if route == "/api/order/sell":
        payload["side"] = "SELL"
        return "place_order"
    return mapping.get(route)


def _build_application() -> tornado.web.Application:
    routes = [
        (r"/health", _GatewayHandler),
        (r"/data/.*", _GatewayHandler),
        (r"/account", _GatewayHandler),
        (r"/positions", _GatewayHandler),
        (r"/orders", _GatewayHandler),
        (r"/trades", _GatewayHandler),
        (r"/order_status", _GatewayHandler),
        (r"/place_order", _GatewayHandler),
        (r"/cancel_order", _GatewayHandler),
        (r"/debug/.*", _GatewayHandler),
        (r"/api/.*", _GatewayHandler),
    ]
    return tornado.web.Application(routes)


def _start_http_server() -> None:
    if RUNTIME.ioloop is not None or RUNTIME.http_thread is not None:
        _emit("warning", "http server already started listen=%s:%s", LISTEN_HOST, LISTEN_PORT)
        return
    RUNTIME.http_started_from = "init"
    if not RUN_HTTP_IN_BACKGROUND_THREAD:
        RUNTIME.direct_dispatch = True
        _emit("info", "starting http server in init thread listen=%s:%s", LISTEN_HOST, LISTEN_PORT)
        _tornado_thread_main(use_current_ioloop=True)
        return
    _start_http_server_background("init")


def _start_http_server_background(reason: str) -> None:
    if RUNTIME.ioloop is not None or RUNTIME.http_thread is not None:
        _emit("warning", "http server already started listen=%s:%s reason=%s", LISTEN_HOST, LISTEN_PORT, reason)
        return
    RUNTIME.http_started_from = reason
    RUNTIME.direct_dispatch = reason == "module_load"
    _emit(
        "info",
        "starting http server in background thread listen=%s:%s reason=%s",
        LISTEN_HOST,
        LISTEN_PORT,
        reason,
    )
    RUNTIME.http_thread = threading.Thread(
        target=_tornado_thread_main,
        name="bt-big-qmt-gateway-http",
        daemon=False,
    )
    RUNTIME.http_thread.start()


def _set_account_on_context(context_info: Any) -> None:
    if _is_placeholder_account_id(ACCOUNT_ID):
        _emit(
            "warning",
            "ACCOUNT_ID is not configured; pass account_id in requests or set ACCOUNT_ID at file top",
        )
        return
    try:
        setter = getattr(context_info, "set_account", None)
        if setter is not None:
            setter(ACCOUNT_ID)
            _emit("info", "ContextInfo.set_account success account=%s", ACCOUNT_ID)
    except Exception as exc:
        _emit("error", "ContextInfo.set_account failed account=%s error=%s", ACCOUNT_ID, exc)
        LOGGER.exception("ContextInfo.set_account failed: %s", exc)
    try:
        context_info.accountID = ACCOUNT_ID
        _emit("info", "ContextInfo.accountID assigned account=%s", ACCOUNT_ID)
    except Exception as exc:
        _emit("warning", "ContextInfo.accountID assign failed account=%s error=%s", ACCOUNT_ID, exc)


def _tornado_thread_main(use_current_ioloop: bool = False) -> None:
    try:
        if use_current_ioloop:
            RUNTIME.ioloop = tornado.ioloop.IOLoop.current()
        else:
            RUNTIME.ioloop = tornado.ioloop.IOLoop()
            try:
                RUNTIME.ioloop.make_current()
            except Exception:
                pass
        app = _build_application()
        RUNTIME.http_server = app.listen(LISTEN_PORT, address=LISTEN_HOST)
        LOGGER.info("bt_big_qmt_gateway tornado listening on %s:%s", LISTEN_HOST, LISTEN_PORT)
        _emit(
            "info",
            "listen success listen=%s:%s direct_dispatch=%s started_from=%s threaded=%s context_ready=%s",
            LISTEN_HOST,
            LISTEN_PORT,
            RUNTIME.direct_dispatch,
            RUNTIME.http_started_from,
            not use_current_ioloop,
            RUNTIME.context_info is not None,
        )
        _emit("info", "entering tornado ioloop; gateway should keep running")
        RUNTIME.ioloop.start()
    except Exception as exc:
        RUNTIME.last_error = "%s\n%s" % (exc, traceback.format_exc())
        LOGGER.exception("bt_big_qmt_gateway tornado thread failed: %s", exc)
        _emit("error", "http server failed listen=%s:%s error=%s", LISTEN_HOST, LISTEN_PORT, exc)
        RUNTIME.http_server = None
        RUNTIME.ioloop = None
        RUNTIME.http_thread = None


def init(ContextInfo):
    RUNTIME.init_called = True
    RUNTIME.context_info = ContextInfo
    _emit(
        "info",
        "init starting account=%s account_type=%s listen=%s:%s background=%s log_file=%s",
        ACCOUNT_ID,
        ACCOUNT_TYPE,
        LISTEN_HOST,
        LISTEN_PORT,
        RUN_HTTP_IN_BACKGROUND_THREAD,
        LOG_FILE,
    )
    _set_account_on_context(ContextInfo)
    LOGGER.info(
        "bt_big_qmt_gateway init starting: account=%s type=%s background=%s log_file=%s",
        ACCOUNT_ID,
        ACCOUNT_TYPE,
        RUN_HTTP_IN_BACKGROUND_THREAD,
        LOG_FILE,
    )
    _start_http_server()
    LOGGER.info("bt_big_qmt_gateway init done: account=%s type=%s", ACCOUNT_ID, ACCOUNT_TYPE)
    _emit("info", "init done account=%s account_type=%s", ACCOUNT_ID, ACCOUNT_TYPE)


def handlebar(ContextInfo):
    RUNTIME.drain(ContextInfo)


def stop(ContextInfo):
    if not STOP_HTTP_ON_QMT_STOP:
        LOGGER.info("bt_big_qmt_gateway stop ignored: STOP_HTTP_ON_QMT_STOP=false")
        _emit("warning", "stop ignored STOP_HTTP_ON_QMT_STOP=false")
        return
    if RUNTIME.ioloop is None:
        _emit("warning", "stop ignored ioloop is none")
        return

    def _shutdown() -> None:
        LOGGER.info("bt_big_qmt_gateway stopping")
        _emit("warning", "http server stopping")
        if RUNTIME.http_server is not None:
            RUNTIME.http_server.stop()
            RUNTIME.http_server = None
        if RUNTIME.ioloop is not None:
            RUNTIME.ioloop.stop()

    RUNTIME.ioloop.add_callback(_shutdown)
    RUNTIME.ioloop = None
    RUNTIME.http_thread = None


def _auto_start_http_on_module_load() -> None:
    if not AUTO_START_HTTP_ON_MODULE_LOAD:
        _emit(
            "info",
            "module autostart disabled; if no later init log appears, QMT did not call init(ContextInfo)",
        )
        return
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        _emit("info", "module autostart skipped in pytest")
        return
    _emit(
        "info",
        "module autostart enabled; HTTP stays alive, data/trading APIs require init(ContextInfo)",
    )
    _start_http_server_background("module_load")


_auto_start_http_on_module_load()
