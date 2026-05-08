import os
import csv
import json
import pickle
import threading
import datetime
import functools
from typing import List, Dict, Set, Optional

import numpy as np
import pandas as pd

from tools.constants import InfoItem, REPURCHASE_CODES
from tools.utils_basic import symbol_to_code, is_in_continuous_auction
from tools.utils_cache_ak import AKCache, AKCacheProtected, TRADE_DAY_CACHE_PATH


trade_day_cache = {}
trade_max_year_key = 'max_year'


# 查询股票名称
class StockNames:
    _instance = None
    _data = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(StockNames, cls).__new__(cls)
            cls._data = None  # Initialize data as None initially
        return cls._instance

    def __init__(self):
        if self._data is None:
            self.load_codes_and_names()

    def load_codes_and_names(self):
        print('[更新缓存] 正在加载股票代码和名称...')
        self._data = get_stock_codes_and_names()
        print('[更新缓存] 加载完毕!')

    def get_code_list(self) -> list:
        return list(self._data.keys())

    def get_name_list(self) -> list:
        return list(self._data.values())

    def add_name(self, code, name) -> bool:
        if self._data is not None:
            self._data[code] = name
            return True
        return False

    def get_name(self, code) -> str:
        if self._data is None:
            self.load_codes_and_names()

        if code in self._data:
            return self._data[code]
        return '[Unknown]'


# ===============
# 获取股票的中文名称
# ===============

CODE_NAME_CACHE_PATH = './_cache/_code_names.csv'


def load_stock_code_and_names(retention_day: int = 1):
    cache_available = False
    df = pd.DataFrame(columns=['代码', '名称', '日期'])

    # 如果有缓存就先Load，然后再看是否过期
    if os.path.exists(CODE_NAME_CACHE_PATH):
        df = pd.read_csv(CODE_NAME_CACHE_PATH, dtype={'代码': str})
        cache_date_str = df['日期'].head(1).values[0]
        cache_date = datetime.datetime.strptime(cache_date_str, '%Y-%m-%d')
        curr_date = datetime.datetime.today()
        if curr_date - cache_date < datetime.timedelta(days=retention_day):
            cache_available = True

    # 过期就尝试下载并缓存新的覆盖旧版本
    if not cache_available:
        try:
            df = AKCache.stock_info_a_code_name()
            df = df.rename(columns={'code': '代码', 'name': '名称'})

            if len(df) == 0:
                df = AKCache.stock_zh_a_spot()  # 这个接口容易封IP，留作备用
                df['代码'] = df['代码'].str[2:]

            df = df[['代码', '名称']]

            try:
                etf_df = AKCache.fund_etf_spot_em()
                etf_df = etf_df[['代码', '名称']]
                df = pd.concat([df, etf_df])
            except Exception as e:
                print('Update remote ETF code and names failed! ', e)

            df = df.sort_values(by='代码')
            df['日期'] = datetime.datetime.today().strftime('%Y-%m-%d')

            df.to_csv(CODE_NAME_CACHE_PATH)
        except Exception as e:
            print('Update remote stock code and names failed! ', e)

    return df


def get_stock_codes_and_names() -> Dict[str, str]:
    ans = {}

    with open('./_cache/_rawdata/mktdt00.txt', 'r', encoding='utf-8', errors='replace') as r:
        lines = r.readlines()
        for line in lines:
            arr = line.split('|')
            if len(arr) > 2 and len(arr[1]) == 6:
                ans[arr[1] + '.SH'] = arr[2]

    with open('./_cache/_rawdata/sjshq.txt', 'r', encoding='utf-8', errors='replace') as r:
        lines = r.readlines()
        for line in lines:
            arr = json.loads(line)
            ans[arr['code']] = arr['name']

    df = load_stock_code_and_names()
    if df is not None:
        df['代码'] = df['代码'].apply(lambda x: symbol_to_code(x))
        ans.update(dict(zip(df['代码'], df['名称'])))
    
    for code in REPURCHASE_CODES:
        ans[code] ='逆回购'
    ans['888880.SH'] = '标准券'
    ans['131990.SZ'] = '标准券'
    return ans


# ================
#  本地磁盘缓存
# ================


def delete_file(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)

        if os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        print(f'delete {path} failed! {str(e)}')


# 读取pickle缓存
def load_pickle(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            loaded_object = pickle.load(f)
        return loaded_object
    else:
        return None


# 存储pickle缓存
def save_pickle(path: str, obj: object) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


# 读取json缓存，如果找不到文件则创建空json并返回
def load_json(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as r:
            ans = r.read()
        return json.loads(ans)
    else:
        with open(path, 'w') as w:
            w.write('{}')
        return {}


# 存储json缓存，全覆盖写入
def save_json(path: str, var: dict, ensure_ascii=True) -> None:
    with open(path, 'w') as w:
        w.write(json.dumps(var, ensure_ascii=ensure_ascii, indent=4))


# 删除json缓存中的单个key-value，key为字符串
def del_key(lock: threading.Lock, path: str, key: str) -> None:
    with lock:
        temp_json = load_json(path)
        if key in temp_json:
            del temp_json[key]
        save_json(path, temp_json)


# 删除json缓存中的多个个key-value，key为字符串
def del_keys(lock: threading.Lock, path: str, keys: List[str]) -> None:
    with lock:
        temp_json = load_json(path)
        for key in keys:
            if key in temp_json:
                del temp_json[key]
        save_json(path, temp_json)


# 清除held持仓，将标记持仓天数为null
def del_held_day(lock: threading.Lock, path: str, key: str):
    with lock:
        try:
            held_info = load_json(path)

            if key in held_info:
                held_info[key][InfoItem.DayCount] = None
            else:
                # 逆回购或者未记录持仓卖出的场景
                held_info[key] = {InfoItem.DayCount: None}

            save_json(path, held_info)
            return True
        except Exception as e:
            print('del held day failed! ', e)
            return False


# 增加新的持仓记录
def new_held(held_operation_lock: threading.Lock, path: str, codes: List[str]) -> None:
    with held_operation_lock:
        held_info = load_json(path)
        for code in codes:
            held_info.update({code: {InfoItem.DayCount: 0}})
            # 等价于：
            # if code in held_info:
            #     held_info[code][InfoItem.DayCount] = 0
            # else:
            #     held_info[code] = {InfoItem.DayCount: 0}
        save_json(path, held_info)


# 所有缓存持仓天数+1，_inc_date为单日判重标记位
def all_held_inc(lock: threading.Lock, path: str) -> bool:
    with lock:
        try:
            held_info = load_json(path)
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            if (InfoItem.IncDate not in held_info) or (held_info[InfoItem.IncDate] != today):
                held_info[InfoItem.IncDate] = today
                for code in held_info.keys():
                    if InfoItem.DayCount not in held_info[code]:
                        continue

                    if held_info[code][InfoItem.DayCount] is None:
                        continue

                    if code != InfoItem.IncDate:
                        held_info[code][InfoItem.DayCount] += 1

                save_json(path, held_info)
                return True
            else:
                return False
        except Exception as e:
            print('[持仓计数] Held days +1 failed! ', e)
            return False


# 更新持仓股买入开始最高价格
def update_max_prices(
    lock: threading.Lock,
    quotes: dict,
    positions: list,
    curr_time: str,
    path_max_prices: str,
    path_min_prices: str,
    path_held_info: str,
    ignore_open_day: bool = True,  # 是否忽略开仓日，从次日开始计算最高价
) -> tuple[dict, dict]:
    held_info = load_json(path_held_info)

    # 只有在连续竞价时间段内才更新最高价格和最低价格
    if not is_in_continuous_auction(curr_time):
        with lock:
            max_prices = load_json(path_max_prices)
        return max_prices, held_info

    with lock:
        max_prices = load_json(path_max_prices)
        min_prices = load_json(path_min_prices)

    max_updated = False
    min_updated = False

    for position in positions:
        code = position.stock_code
        if code in held_info:  # 只更新持仓超过一天的
            if ignore_open_day:  # 忽略开仓日的最高价
                if InfoItem.DayCount not in held_info[code]:
                    continue

                held_day = held_info[code][InfoItem.DayCount]
                if held_day is None or held_day <= 0:
                    continue

            if code in quotes:
                quote = quotes[code]

                # 更新历史最高
                high_price = quote['high']
                if code in max_prices:
                    if max_prices[code] < high_price:
                        max_prices[code] = round(high_price, 3)
                        max_updated = True
                else:
                    max_prices[code] = round(high_price, 3)
                    max_updated = True

                # 更新历史最低
                low_price = quote['low']
                if code in min_prices:
                    if min_prices[code] > low_price:
                        min_prices[code] = round(low_price, 3)
                        min_updated = True
                else:
                    min_prices[code] = round(low_price, 3)
                    min_updated = True

    if max_updated:
        with lock:
            save_json(path_max_prices, max_prices)

    if min_updated:
        with lock:
            save_json(path_min_prices, min_prices)

    return max_prices, held_info


# 获取磁盘文件中的symbol列表，假设一行是一个symbol
def load_symbols(path: str) -> list[str]:
    if os.path.exists(path):
        with open(path, 'r') as r:
            symbols = r.read().split('\n')
        return symbols
    else:
        return []


# symbol列表存储到磁盘文件中，假设一行是一个symbol
def save_symbols(path: str, symbols: list[str]) -> None:
    with open(path, 'w') as w:
        w.write('\n'.join(symbols))


# 记录成交单
def record_deal(
    lock: threading.Lock,
    path: str,
    timestamp: str,
    code: str,
    name: str,
    order_type: str,
    remark: str,
    price: float,
    volume: int,
):
    with lock:
        if not os.path.exists(path):
            with open(path, 'w') as w:
                w.write(','.join(['日期', '时间', '代码', '名称', '类型', '注释', '成交价', '成交量']))
                w.write('\n')

        with open(path, 'a+', newline='') as w:
            wf = csv.writer(w)
            dt = datetime.datetime.fromtimestamp(int(timestamp))

            wf.writerow([
                dt.date(), dt.time(),
                code, name, order_type, remark, price, volume
            ])


# ================
#  交易日历缓存
# ================


# 获取磁盘缓存的交易日列表
def get_disk_trade_day_list_and_update_max_year() -> list:
    # 读磁盘，这里可以有内存缓存的速度优化
    df = pd.read_csv(TRADE_DAY_CACHE_PATH)
    trade_dates = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d').values
    trade_day_cache[trade_max_year_key] = trade_dates[-1][:4]
    return trade_dates


# 获取前n个交易日，返回格式 基本格式：%Y%m%d，扩展格式：%Y-%m-%d
# 如果为非交易日，则取上一个交易日为前0天
def get_prev_trading_date(now: datetime.datetime, count: int, basic_format: bool = True) -> str:
    today = now.strftime('%Y-%m-%d')
    return get_prev_trading_date_str(today, count, basic_format)


@functools.cache
def get_prev_trading_date_str(today: str, count: int, basic_format: bool = True) -> str:
    trading_day_list = get_disk_trade_day_list_and_update_max_year()
    if len(today) == 8:
        today = f'{today[0:4]}-{today[4:6]}-{today[6:8]}'

    try:
        trading_index = list(trading_day_list).index(today)
    except ValueError:
        trading_index = np.searchsorted(trading_day_list, today) - 1

    if trading_index + count >= 0:
        if basic_format:
            return trading_day_list[trading_index - count].replace('-', '')
        else:
            return trading_day_list[trading_index - count]
    else:
        print('[CACHE] 找不到目标，默认返回已知最早的交易日')
        if basic_format:
            return trading_day_list[0].replace('-', '')
        else:
            return trading_day_list[0]


# 获取后n个交易日，返回格式 基本格式：%Y%m%d，扩展格式：%Y-%m-%d
# 如果为非交易日，则取下一个交易日为后0天
def get_next_trading_date(now: datetime.datetime, count: int, basic_format: bool = True) -> str:
    today = now.strftime('%Y-%m-%d')
    return get_next_trading_date_str(today, count, basic_format)


@functools.cache
def get_next_trading_date_str(today: str, count: int, basic_format: bool = True) -> str:
    trading_day_list = get_disk_trade_day_list_and_update_max_year()

    if len(today) == 8:
        today = f'{today[0:4]}-{today[4:6]}-{today[6:8]}'

    try:
        trading_index = list(trading_day_list).index(today)
    except ValueError:
        trading_index = np.searchsorted(trading_day_list, today) - 1

    if trading_index + count < len(trading_day_list):
        if basic_format:
            return trading_day_list[trading_index + count].replace('-', '')
        else:
            return trading_day_list[trading_index + count]
    else:
        print('[CACHE] 找不到目标，默认返回已知最晚的交易日')
        if basic_format:
            return trading_day_list[-1].replace('-', '')
        else:
            return trading_day_list[-1]


# 获取前n个交易日列表，返回格式 %Y-%m-%d
@functools.cache
def get_prev_trading_date_list(today: str, count: int) -> list:
    if len(today) == 8:
        today = f'{today[0:4]}-{today[4:6]}-{today[6:8]}'

    trading_day_list = get_disk_trade_day_list_and_update_max_year()
    try:
        trading_index = list(trading_day_list).index(today)
    except ValueError:
        trading_index = np.searchsorted(trading_day_list, today) - 1
    
    return trading_day_list[trading_index - count : trading_index]


# 获取从 start_day 到 end_day 的交易日列表，返回列表，其中日期格式 %Y-%m-%d
@functools.cache
def get_trading_date_list(start_date: str, end_date: str) -> list:
    if start_date == end_date:
        return [start_date]
    if len(start_date) == 8:
        start_date = f'{start_date[0:4]}-{start_date[4:6]}-{start_date[6:8]}'
    if len(end_date) == 8:
        end_date = f'{end_date[0:4]}-{end_date[4:6]}-{end_date[6:8]}'
        
    trading_day_list = get_disk_trade_day_list_and_update_max_year()
    try:
        start_trading_index = list(trading_day_list).index(start_date)
    except ValueError:
        start_trading_index = np.searchsorted(trading_day_list, start_date)
    
    if start_date > end_date:
        return trading_day_list[start_trading_index : start_trading_index+1]
    
    try:
        end_trading_index = list(trading_day_list).index(end_date)
    except ValueError:
        end_trading_index = np.searchsorted(trading_day_list, end_date) - 1
    
    if end_trading_index < len(trading_day_list):
        return trading_day_list[start_trading_index : end_trading_index + 1]
    else:
        print('[CACHE] 找不到目标，默认返回已知最晚的交易日')
        return trading_day_list[start_trading_index : -1]


# 检查当日是否是交易日，使用sina数据源
def check_is_open_day_sina(curr_date: str) -> bool:
    """
    curr_date example: '2024-12-31'
    """
    curr_year = curr_date[:4]

    # 内存缓存
    if curr_date in trade_day_cache:
        if curr_year <= trade_day_cache[trade_max_year_key]:
            return trade_day_cache[curr_date]

    # 文件缓存
    if os.path.exists(TRADE_DAY_CACHE_PATH):  # 文件缓存存在
        trade_day_list = get_disk_trade_day_list_and_update_max_year()
        if curr_year <= trade_day_cache[trade_max_year_key]:  # 未过期
            ans = curr_date in trade_day_list
            trade_day_cache[curr_date] = ans
            print(f'[文件缓存] {curr_date} 为 {"" if ans else "非"}交易日')
            return ans

    # 网络缓存（ttl=0 的 AK 封装：每次尝试拉取，失败则读 TRADE_DAY_CACHE_PATH 过期 CSV）
    df = AKCache.tool_trade_date_hist_sina()
    if df is None or len(df) == 0:
        print('[网络缓存] 交易日历拉取失败且无可用缓存')
        return True
    print(f'[网络缓存] 更新交易日历 {curr_year} - {int(curr_year) + 1} 已存入 {TRADE_DAY_CACHE_PATH}.')

    trade_day_list = get_disk_trade_day_list_and_update_max_year()
    if curr_year <= trade_day_cache[trade_max_year_key]:  # 未过期
        ans = curr_date in trade_day_list
        trade_day_cache[curr_date] = ans
        print(f'[网络缓存] {curr_date} 为 {"" if ans else "非"}交易日')
        return ans

    # 实在拿不到数据默认为True
    print(f'[DO NOT KNOW {curr_date}, default to True trade day]')
    return True


def check_is_open_day(curr_date: str) -> bool:
    """
    curr_date example: '2024-12-31'
    """
    return check_is_open_day_sina(curr_date)


# 装饰器：检查是否是交易日，非交易日不执行函数
def check_open_day(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return func(*args, **kwargs)    # 开放日正常执行
    return wrapper


# ================
#  远程数据缓存
# ================


def _index_constituent_df_ok(df: Optional[pd.DataFrame]) -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
        return False
    return '品种代码' in df.columns or '成分券代码' in df.columns


def _load_legacy_index_constituent_pkl(index_symbol: str) -> Optional[pd.DataFrame]:
    legacy = f'./_cache/_index_{index_symbol}.pkl'
    if not os.path.exists(legacy):
        return None
    try:
        return pd.read_pickle(legacy)
    except Exception:
        return None


# 获取指数成份symbol
def get_index_constituent_symbols(index_symbol: str) -> list[str]:
    if index_symbol[:2] in ['00', '93', '89']:
        df = AKCache.index_stock_cons_csindex(symbol=index_symbol)
        if not _index_constituent_df_ok(df):
            legacy = _load_legacy_index_constituent_pkl(index_symbol)
            if _index_constituent_df_ok(legacy):
                df = legacy
        if not _index_constituent_df_ok(df):
            df = AKCache.index_stock_cons(symbol=index_symbol)
            print('警告：指数成份使用备用数据，监控股池可能不完整，请注意！')
    else:
        df = AKCache.index_stock_cons(symbol=index_symbol)
        print('警告：指数成份使用备用数据，监控股池可能不完整，请注意！')

    if '品种代码' in df.columns:
        return [str(code).zfill(6) for code in df['品种代码'].values]
    else:
        return [str(code).zfill(6) for code in df['成分券代码'].values]


# 获取指数成份code
def get_index_constituent_codes(index_symbol: str) -> list:
    symbols = get_index_constituent_symbols(index_symbol)
    return [symbol_to_code(str(symbol).zfill(6)) for symbol in symbols]


# 获取市值符合范围的code列表
def get_market_value_limited_codes(code_prefixes: Set[str], min_value: int, max_value: int) -> list[str]:
    df = AKCache.stock_zh_a_spot_em()
    df = df.sort_values('代码')
    df = df[['代码', '名称', '总市值', '流通市值']]
    df = df[(min_value < df['总市值']) & (df['总市值'] < max_value)]
    df = df[df['代码'].str.startswith(tuple(code_prefixes))]
    return [symbol_to_code(symbol) for symbol in df['代码'].to_list()]


# 获取当日可用的股票代码
def get_available_stock_codes() -> list[str]:
    df = AKCache.stock_info_a_code_name()
    codes = [symbol_to_code(symbol) for symbol in df['code'].values]
    return list(set(codes))


def _filter_none_st_out(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df['name'].str.contains('ST')]
    df = df[~df['name'].str.endswith('退')]
    df = df[~df['name'].str.startswith('退市')]
    return df


# 以下三处单独使用 AKCacheProtected：AK 故障时用过期 CSV 兜底，避免选股/白名单股池为空（其余逻辑仍用 AKCache）
# 根据两位数前缀获取股票列表
def get_prefixes_stock_codes(prefixes: Set[str], none_st: bool = False) -> List[str]:
    """
    prefixes: 六位数的两位数前缀
    """
    df = AKCacheProtected.stock_info_a_code_name()
    if none_st:
        df = _filter_none_st_out(df)
    codes = [symbol_to_code(symbol) for symbol in df['code'].values if symbol[:2] in prefixes]
    return list(set(codes))


def get_none_st_codes() -> list[str]:
    df = AKCacheProtected.stock_info_a_code_name()
    df = _filter_none_st_out(df)
    codes = [symbol_to_code(symbol) for symbol in df['code'].values]
    return list(set(codes))


# 获取流通市值，单位（元）
def get_stock_codes_and_circulation_mv() -> Dict[str, int]:
    df = AKCacheProtected.stock_zh_a_spot_em()
    df['代码'] = df['代码'].apply(lambda x: symbol_to_code(x))
    df = df[['代码', '流通市值']].dropna()
    return dict(zip(df['代码'], df['流通市值']))


# ===============
# 获取股票的分红配送
# ===============


def _get_recent_fhps_report_date(now: Optional[datetime.datetime] = None) -> str:
    now = now or datetime.datetime.now()
    year = now.year
    month = now.month

    # 只调用一次接口时，优先选择当前时点最可能覆盖最近除权事件的最近报告期。
    if month <= 6:
        return f'{year - 1}1231'
    if month <= 9:
        return f'{year}0630'
    return f'{year}0930'


def get_recent_exit_right_codes_from_fhps(
    days: int,
    now: Optional[datetime.datetime] = None,
    report_date: Optional[str] = None,
) -> List[str]:
    if days <= 0:
        return []

    now = now or datetime.datetime.now()
    report_date = report_date or _get_recent_fhps_report_date(now)
    df = AKCache.stock_fhps_em(date=report_date)
    if df is None or len(df) == 0:
        return []

    if '代码' not in df.columns or '除权除息日' not in df.columns:
        print(f'[分红送配] {report_date} 返回结果缺少关键列，跳过')
        return []

    target_dates = {
        get_prev_trading_date(now, forward_day)
        for forward_day in range(days - 1, -1, -1)
    }

    df = df.copy()
    df['除权除息日'] = pd.to_datetime(df['除权除息日'], errors='coerce').dt.strftime('%Y%m%d')
    df = df[df['除权除息日'].isin(target_dates)]

    codes = [
        symbol_to_code(symbol)
        for symbol in df['代码'].astype(str).tolist()
        if len(str(symbol)) == 6
    ]
    return sorted(set(codes))
