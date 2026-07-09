"""
QMT xtquant / xtdata 远程数据：板块常量、日线历史等。

板块列表见 XtSectorType；日线批量接口见 get_qmt_daily_histories / get_qmt_daily_history。
"""
import logging
import threading
import time
from typing import Any, Callable

import pandas as pd

from tools.constants import ExitRight

logger = logging.getLogger(__name__)

_DAILY_FIELDS = ("time", "open", "high", "low", "close", "volume", "amount")
# 单股 download_history_data 超时（沿用原 miniqmt 行为）
_QMT_SINGLE_DOWNLOAD_TIMEOUT = 3.0
# download_history_data2 分批大小；过大时 QMT 回调/内存压力更高
_QMT_BATCH_DOWNLOAD_SIZE = 100
# 每批 download_history_data2 最长等待（秒）；超时则回退单股下载
_QMT_BATCH_DOWNLOAD_TIMEOUT = 90.0
# 至少 2 只股票才走 batch2，单股仍用同步 download_history_data
_QMT_BATCH_MIN_COUNT = 2


class XtSectorType:
    HSA_STOCK = "沪深京A股"   # 上交所A股、深交所A股和北交所的股票列表
    HS_STOCK = "沪深A股"      # 上交所A股和深交所A股的股票列表
    SH_STOCK = "上证A股"      # 上交所A股的股票列表
    SZ_STOCK = "深证A股"      # 深交所A股的股票列表
    BJ_STOCK = "京市A股"      # 北交所的股票列表

    HS_INDEX = "沪深指数"     # 上交所和深交所指数列表
    SH_INDEX = "沪市指数"     # 上交所指数列表
    SZ_INDEX = "深市指数"     # 深交所指数列表

    HS_ETF = "沪深ETF"        # 上交所、深交所的 ETF 列表
    SH_ETF = "沪市ETF"        # 上交所的 ETF 列表
    SZ_ETF = "深市ETF"        # 深交所的 ETF 列表

    HS_KZZ = "沪深转债"       # 上交所、深交所的可转债列表
    SH_KZZ = "上证转债"       # 上交所的可转债列表
    SZ_KZZ = "深证转债"       # 深交所的可转债列表

    HS_HKT = "香港联交所股票"  # 沪深港通标的（QMT 以联交所股票板块聚合）

    HS_GLRA = "沪深债券"      # 沪深逆回购等债券品种
    SH_GLRA = "沪市债券"      # 上交所逆回购
    SZ_GLRA = "深市债券"      # 深交所逆回购


class XtDividendType:
    NONE = "none"                 # 不复权
    FRONT = "front"               # 前复权
    BACK = "back"                 # 后复权
    FRONT_RATIO = "front_ratio"   # 等比前复权
    BACK_RATIO = "back_ratio"     # 等比后复权


def parse_xt_sectors(raw: str, default: tuple[str, ...]) -> list[str]:
    """解析逗号分隔的板块；支持 XtSectorType 成员名或板块名字符串。"""
    sectors: list[str] = []
    seen: set[str] = set()
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        sector = getattr(XtSectorType, name, name)
        if sector not in seen:
            seen.add(sector)
            sectors.append(sector)
    return sectors or list(default)


def _run_with_timeout(func: Callable[..., Any], args: tuple[Any, ...], timeout: float) -> Any:
    """兜底：xtdata 下载接口无原生 timeout，社区常用 threading 包一层。

    注意：超时后底层 C++ 调用仍在 daemon 线程里跑，无法强杀，只是主流程不再等。
    """
    result: list[Any] = [None]
    error: list[BaseException | None] = [None]

    def _wrapper() -> None:
        try:
            result[0] = func(*args)
        except BaseException as exc:
            error[0] = exc

    thread = threading.Thread(target=_wrapper, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise Exception(f"函数执行超过 {timeout} 秒，触发超时")
    if error[0] is not None:
        raise error[0]
    return result[0]


def _get_xtdata():
    from xtquant import xtdata

    xtdata.enable_hello = False
    return xtdata


def _exit_right_to_dividend_type(adjust: ExitRight) -> str:
    if adjust == ExitRight.QFQ:
        return XtDividendType.FRONT
    if adjust == ExitRight.HFQ:
        return XtDividendType.BACK
    return XtDividendType.NONE


def _peek_code_daily(
    code: str,
    start_time: str,
    end_time: str,
    dividend_type: str,
) -> pd.DataFrame:
    """读本地缓存日线（get_market_data_ex），不触发下载。"""
    xtdata = _get_xtdata()
    raw = xtdata.get_market_data_ex(
        field_list=list(_DAILY_FIELDS),
        stock_list=[code],
        period="1d",
        start_time=start_time,
        end_time=end_time,
        count=-1,
        dividend_type=dividend_type,
        fill_data=False,
    )
    df = raw.get(code) if isinstance(raw, dict) else None
    if df is None or df.empty:
        return pd.DataFrame()
    if "time" not in df.columns:
        df = df.reset_index(names="time")
    return df


def _daily_history_covers_range(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
) -> bool:
    if df.empty or "time" not in df.columns:
        return False
    if not start_time and not end_time:
        return True

    times = pd.to_numeric(df["time"], errors="coerce").dropna()
    if times.empty:
        return False

    min_time = int(times.min())
    max_time = int(times.max())
    if start_time and min_time > int(start_time):
        return False
    if end_time and max_time < int(end_time):
        return False
    return True


def _code_cache_ready(code: str, start_time: str, end_time: str, dividend_type: str) -> bool:
    cached = _peek_code_daily(code, start_time, end_time, dividend_type)
    return _daily_history_covers_range(cached, start_time, end_time)


def _download_code_daily(code: str, start_time: str, end_time: str, incrementally: bool = False) -> None:
    """单股下载未复权原始数据到本地；incrementally 或 start_time 为空时走增量补数。"""
    xtdata = _get_xtdata()
    if incrementally or not start_time:
        xtdata.download_history_data(
            stock_code=code,
            period="1d",
            start_time="",
            end_time=end_time,
            incrementally=True,
        )
        return

    xtdata.download_history_data(
        stock_code=code,
        period="1d",
        start_time=start_time,
        end_time=end_time,
    )


def _invoke_download_history_data2(
    xtdata: Any,
    stock_list: list[str],
    start_time: str,
    end_time: str,
    callback: Callable[[dict], None],
    incrementally: bool = False,
) -> None:
    """兼容不同 xtquant 版本的 download_history_data2 参数形式。"""
    dl_start = "" if incrementally else start_time
    kwargs = {
        "stock_list": stock_list,
        "period": "1d",
        "start_time": dl_start,
        "end_time": end_time,
        "callback": callback,
    }
    if incrementally:
        kwargs["incrementally"] = True
    try:
        xtdata.download_history_data2(**kwargs)
    except TypeError:
        if incrementally:
            xtdata.download_history_data2(
                stock_list,
                period="1d",
                start_time=dl_start,
                end_time=end_time,
                callback=callback,
                incrementally=True,
            )
        else:
            xtdata.download_history_data2(
                stock_list,
                period="1d",
                start_time=dl_start,
                end_time=end_time,
                callback=callback,
            )


def _chunk_codes_cached(
    codes: list[str],
    start_time: str,
    end_time: str,
    dividend_type: str,
) -> bool:
    if not codes:
        return True
    return all(_code_cache_ready(code, start_time, end_time, dividend_type) for code in codes)


def _wait_batch_download2(
    codes: list[str],
    start_time: str,
    end_time: str,
    dividend_type: str,
    timeout: float,
    incrementally: bool = False,
) -> bool:
    """等待 download_history_data2 完成；回调 finished==total 或本地缓存齐备即视为成功。"""
    progress = {"finished": 0, "total": 0, "done": False, "error": None}
    lock = threading.Lock()

    def callback(data: dict) -> None:
        if not isinstance(data, dict):
            return
        with lock:
            progress["finished"] = data.get("finished", progress["finished"])
            progress["total"] = data.get("total", progress["total"])
            total = progress["total"]
            if total > 0 and progress["finished"] >= total:
                progress["done"] = True

    def worker() -> None:
        try:
            xtdata = _get_xtdata()
            _invoke_download_history_data2(
                xtdata,
                list(codes),
                start_time,
                end_time,
                callback,
                incrementally=incrementally,
            )
        except BaseException as exc:
            with lock:
                progress["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with lock:
            if progress["done"]:
                return True
            if progress["error"] is not None:
                logger.warning("download_history_data2 error: %s", progress["error"])
                return False
        if _chunk_codes_cached(codes, start_time, end_time, dividend_type):
            return True
        time.sleep(0.2)

    logger.warning(
        "download_history_data2 timeout (%ss) for %d codes (%s-%s)",
        timeout,
        len(codes),
        start_time or "*",
        end_time or "*",
    )
    return False


def _download_codes_batch2(
    codes: list[str],
    start_time: str,
    end_time: str,
    dividend_type: str,
    incrementally: bool = False,
) -> list[str]:
    """批量下载；返回仍需单股回退的代码列表。"""
    if len(codes) < _QMT_BATCH_MIN_COUNT:
        return list(codes)

    fallback: list[str] = []
    for i in range(0, len(codes), _QMT_BATCH_DOWNLOAD_SIZE):
        chunk = codes[i:i + _QMT_BATCH_DOWNLOAD_SIZE]
        if _chunk_codes_cached(chunk, start_time, end_time, dividend_type):
            continue
        if _wait_batch_download2(
            chunk,
            start_time,
            end_time,
            dividend_type,
            _QMT_BATCH_DOWNLOAD_TIMEOUT,
            incrementally=incrementally,
        ):
            if not _chunk_codes_cached(chunk, start_time, end_time, dividend_type):
                logger.warning(
                    "download_history_data2 finished but cache still incomplete for chunk size %d",
                    len(chunk),
                )
                fallback.extend(chunk)
            continue
        fallback.extend(chunk)
    return fallback


def _download_codes_single(
    codes: list[str],
    start_time: str,
    end_time: str,
    dividend_type: str,
    incrementally: bool = False,
) -> list[str]:
    """单股 download_history_data + 超时兜底（原 miniqmt 行为）。"""
    fetched: list[str] = []
    for code in codes:
        if _code_cache_ready(code, start_time, end_time, dividend_type):
            fetched.append(code)
            continue
        try:
            _run_with_timeout(
                _download_code_daily,
                (code, start_time, end_time, incrementally),
                _QMT_SINGLE_DOWNLOAD_TIMEOUT,
            )
            fetched.append(code)
        except Exception as e:
            print(f"{code}:下载{e}")
    return fetched


def _ensure_codes_downloaded(
    code_list: list[str],
    start_time: str,
    end_time: str,
    dividend_type: str,
    incrementally: bool = False,
) -> list[str]:
    """先读后下：缓存齐备则跳过；多股 batch2；失败回退单股。"""
    need_download: list[str] = []
    ready: list[str] = []

    for code in code_list:
        if _code_cache_ready(code, start_time, end_time, dividend_type):
            ready.append(code)
        else:
            need_download.append(code)

    if not need_download:
        return ready

    fallback = _download_codes_batch2(
        need_download, start_time, end_time, dividend_type, incrementally=incrementally,
    )
    single_fetched = _download_codes_single(
        fallback, start_time, end_time, dividend_type, incrementally=incrementally,
    )

    fetched_set = set(ready) | set(single_fetched)
    for code in need_download:
        if _code_cache_ready(code, start_time, end_time, dividend_type):
            fetched_set.add(code)

    return [code for code in code_list if code in fetched_set]


def _download_and_fetch_qmt_daily(code_list: list[str], start_time: str, end_time: str, adjust: ExitRight) -> dict:
    xtdata = _get_xtdata()
    period = "1d"
    dividend_type = _exit_right_to_dividend_type(adjust)

    fetched_codes = _ensure_codes_downloaded(
        code_list, start_time, end_time, dividend_type, incrementally=False,
    )
    if not fetched_codes:
        return {}

    data = xtdata.get_market_data(
        field_list=list(_DAILY_FIELDS),
        stock_list=fetched_codes,
        period=period,
        start_time=start_time,
        end_time=end_time,
        count=-1,
        dividend_type=dividend_type,
        fill_data=False,
    )
    return data


def _qmt_to_standard(input_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not input_dict:
        return pd.DataFrame()

    processed_dfs = []

    for col_name, df in input_dict.items():
        transposed = df.transpose()
        transposed = transposed.reset_index().rename(columns={"index": "date_str"})
        long_format = transposed.melt(
            id_vars=["date_str"],
            var_name="code",
            value_name=col_name,
        )
        processed_dfs.append(long_format)

    merged_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=["date_str", "code"], how="outer")

    merged_df["datetime"] = merged_df["date_str"].astype(int)
    merged_df = merged_df.drop("date_str", axis=1)
    merged_df = merged_df.rename(columns={"time": "timestamp"})
    merged_df[["open", "close", "high", "low"]] = merged_df[["open", "close", "high", "low"]].round(2)
    merged_df["amount"] = merged_df["amount"].round(2)

    column_order = ["code", "datetime"] + [
        col for col in merged_df.columns
        if col not in ["code", "datetime"]
    ]
    merged_df = merged_df[column_order]

    return merged_df


def ensure_qmt_daily_downloaded(
    code_list: list[str],
    start_time: str,
    end_time: str,
    incrementally: bool = False,
) -> list[str]:
    """仅下载到 QMT 本地缓存（先读后下 + batch2 + 单股回退）。返回本地可读代码列表。"""
    dividend_type = XtDividendType.FRONT
    return _ensure_codes_downloaded(
        code_list,
        start_time,
        end_time,
        dividend_type,
        incrementally=incrementally,
    )


def get_qmt_daily_histories(
    code_list: list[str],
    start_time: str,
    end_time: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> pd.DataFrame:
    data = _download_and_fetch_qmt_daily(code_list, start_time, end_time, adjust)
    if not data:
        return None

    df = _qmt_to_standard(data)
    if df is not None and len(df) > 0:
        if columns is not None:
            return df[columns]
        return df
    return None


def get_qmt_daily_history(
    code: str,
    start_time: str,
    end_time: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> pd.DataFrame:
    return get_qmt_daily_histories([code], start_time, end_time, columns, adjust)
