"""
AkShare 请求与本地 CSV 缓存（TTL + 可选过期兜底）。
akshare 在首次真正发起请求时才 import，避免「只用到 utils_cache 里其它工具」的环境硬依赖 akshare。
"""
import os
import datetime
import functools
import pandas as pd
from typing import Any, Callable, Optional


AKSHARE_API_CACHE_DIR = './_cache/_api_akshare'
TRADE_DAY_CACHE_PATH = f'{AKSHARE_API_CACHE_DIR}/_open_day_list_sina.csv'


@functools.cache
def _ak_module():
    import akshare as ak
    return ak


def cache_with_path_ttl(path: str | Callable[..., str], ttl: int, dtype: dict) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_path = path(*args, **kwargs) if callable(path) else path
            dir_name = os.path.dirname(cache_path)
            try:
                with open(cache_path, 'rb') as f:
                    if datetime.datetime.now().timestamp() - os.fstat(f.fileno()).st_mtime < ttl:
                        return pd.read_csv(f, dtype=dtype)
            except FileNotFoundError:
                os.makedirs(dir_name, exist_ok=True) if dir_name else None
            try:
                data = func(*args, **kwargs)
                if data is not None:
                    data.to_csv(cache_path, index=False)
                return data
            except Exception as e:
                print('AKShare request failed: ', e)
                return None

        return wrapper

    return decorator


def _read_csv_disk_ignore_ttl(cache_path: str, dtype: dict) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(cache_path, dtype=dtype)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return None
    if df is None or len(df) == 0:
        return None
    return df


def cache_with_path_ttl_protected(path: str | Callable[..., str], ttl: int, dtype: dict) -> Callable:
    """
    与 cache_with_path_ttl 相同的新鲜度策略；在请求失败或返回空表时，
    若磁盘上已有同名缓存文件（即使已过期），则返回该内容而不是 None。
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache_path = path(*args, **kwargs) if callable(path) else path
            dir_name = os.path.dirname(cache_path)
            try:
                with open(cache_path, 'rb') as f:
                    if datetime.datetime.now().timestamp() - os.fstat(f.fileno()).st_mtime < ttl:
                        return pd.read_csv(f, dtype=dtype)
            except FileNotFoundError:
                os.makedirs(dir_name, exist_ok=True) if dir_name else None

            stale = _read_csv_disk_ignore_ttl(cache_path, dtype)

            try:
                data = func(*args, **kwargs)
            except Exception as e:
                print('AKShare request failed: ', e)
                if stale is not None:
                    print(f'[AKCacheProtected·过期兜底] 网络请求失败，已回读磁盘缓存: {cache_path}')
                    return stale
                return None

            if isinstance(data, pd.DataFrame) and len(data) == 0:
                if stale is not None:
                    print(f'[AKCacheProtected·过期兜底] 接口返回空表，已回读磁盘缓存: {cache_path}')
                    return stale
                return data

            if data is not None:
                data.to_csv(cache_path, index=False)
            return data

        return wrapper

    return decorator


# 每项: (akshare 函数名, 缓存路径或 path(cls,...) 的 lambda, ttl 秒, read_csv 的 dtype[, 变体标记])
# 变体 'fhps_date' -> 方法 (cls, date)；'index_symbol' -> 方法 (cls, symbol)。
_AK_CACHE_SPECS = [
    # A 股代码与简称全表；股池、名称映射、过滤 ST、load_stock_code_and_names 等
    ('stock_info_a_code_name', f'{AKSHARE_API_CACHE_DIR}/stock_info_a_code_name.csv', 60 * 60 * 23, {'code': str}),
    # 场内 ETF 实时列表；合并进股票代码名称缓存等
    ('fund_etf_spot_em', f'{AKSHARE_API_CACHE_DIR}/fund_etf_spot_em.csv', 60 * 60 * 23, {'代码': str}),
    # 东财 A 股实时行情（全市场）；市值筛选、流通市值字典等，TTL 短防封
    ('stock_zh_a_spot_em', f'{AKSHARE_API_CACHE_DIR}/stock_zh_a_spot_em.csv', 5, {'代码': str}),
    # 新浪 A 股实时行情；stock_info 为空时的备用代码表，TTL 短
    ('stock_zh_a_spot', f'{AKSHARE_API_CACHE_DIR}/stock_zh_a_spot.csv', 5, {'代码': str}),
    # 新浪交易日历；ttl=0 表示装饰器不凭 mtime 跳过请求，由 utils_cache 上层判断是否要拉新，失败则读本文件过期数据
    ('tool_trade_date_hist_sina', TRADE_DAY_CACHE_PATH, 0, {'trade_date': str}),
    (
        # 中证指数官网成份；00/93/89 开头指数优先数据源
        'index_stock_cons_csindex',
        lambda _cls, symbol: f'{AKSHARE_API_CACHE_DIR}/index_stock_cons_csindex_{symbol}.csv',
        60 * 60 * 23,
        {},
        'index_symbol',
    ),
    (
        # 东财通用指数成份；非中证指数或 csindex 失败时的备用
        'index_stock_cons',
        lambda _cls, symbol: f'{AKSHARE_API_CACHE_DIR}/index_stock_cons_{symbol}.csv',
        60 * 60 * 23,
        {},
        'index_symbol',
    ),
    (
        # 分红送配（报告期）；除权日筛选、最近除权代码等
        'stock_fhps_em',
        lambda _cls, date: f'{AKSHARE_API_CACHE_DIR}/stock_fhps_em_{date}.csv',
        60 * 60 * 23,
        {'代码': str},
        'fhps_date',
    ),
]


def _ak_fetch_noarg(method_name: str) -> Callable[..., Any]:
    def fetch(cls: Any) -> Any:
        return getattr(_ak_module(), method_name)()

    fetch.__name__ = method_name
    return fetch


def _ak_fetch_fhps_date(method_name: str) -> Callable[..., Any]:
    def fetch(cls: Any, date: str) -> Any:
        return getattr(_ak_module(), method_name)(date=date)

    fetch.__name__ = method_name
    return fetch


def _ak_fetch_index_symbol(method_name: str) -> Callable[..., Any]:
    def fetch(cls: Any, symbol: str) -> Any:
        return getattr(_ak_module(), method_name)(symbol=symbol)

    fetch.__name__ = method_name
    return fetch


def _build_ak_cache_type(type_name: str, *, stale_fallback: bool) -> type:
    dec = cache_with_path_ttl_protected if stale_fallback else cache_with_path_ttl
    namespace: dict[str, Any] = {'__module__': __name__, '__qualname__': type_name}
    for spec in _AK_CACHE_SPECS:
        if len(spec) == 5:
            method_name, path, ttl, dtype, variant = spec[0], spec[1], spec[2], spec[3], spec[4]
            if variant == 'fhps_date':
                raw = _ak_fetch_fhps_date(method_name)
            elif variant == 'index_symbol':
                raw = _ak_fetch_index_symbol(method_name)
            else:
                raise ValueError(f'Unknown _AK_CACHE_SPECS variant: {variant!r}')
        else:
            method_name, path, ttl, dtype = spec[0], spec[1], spec[2], spec[3]
            raw = _ak_fetch_noarg(method_name)
        raw.__qualname__ = f'{type_name}.{method_name}'
        namespace[method_name] = classmethod(dec(path=path, ttl=ttl, dtype=dtype)(raw))
    return type(type_name, (), namespace)


AKCache = _build_ak_cache_type('AKCache', stale_fallback=False)
AKCacheProtected = _build_ak_cache_type('AKCacheProtected', stale_fallback=True)
AKCacheProtected.__doc__ = (
    '与 AKCache 相同的接口与缓存路径；在请求失败或返回空表时，'
    '回读磁盘上已存在的过期缓存，尽量避免返回 None。'
)
