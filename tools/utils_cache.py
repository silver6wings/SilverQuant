import os
import csv
import json
import pickle
import threading
import datetime
from typing import List, Dict, Set, Optional

import numpy as np
import pandas as pd
import akshare as ak

from tools.utils_basic import symbol_to_code

trade_day_cache = {}
trade_max_year_key = 'max_year'
TRADE_DAY_CACHE_PATH = './_cache/_open_day_list_sina.csv'
CODE_NAME_CACHE_PATH = './_cache/_code_names.csv'


# 指数常量
class IndexSymbol:
    INDEX_SH_ZS = '000001'      # 上证指数
    INDEX_SH_50 = '000016'      # 上证50
    INDEX_SZ_CZ = '399001'      # 深证指数
    INDEX_SZ_50 = '399850'      # 深证50
    INDEX_SZ_100 = '399330'     # 深证100
    INDEX_HS_300 = '000300'     # 沪深300
    INDEX_ZZ_100 = '000903'     # 中证100
    INDEX_ZZ_500 = '000905'     # 中证500
    INDEX_ZZ_800 = '000906'     # 中证800
    INDEX_ZZ_1000 = '000852'    # 中证1000
    INDEX_ZZ_2000 = '932000'    # 中证2000
    INDEX_ZZ_ALL = '000985'     # 中证全指
    INDEX_CY_ZS = '399006'      # 创业指数
    INDEX_KC_50 = '000688'      # 科创50
    INDEX_BZ_50 = '899050'      # 北证50
    INDEX_ZX_100 = '399005'     # 中小100
    INDEX_ZZ_A50 = '000050'     # 中证A50
    INDEX_ZZ_A500 = '000510'    # 中证A500


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
        print('Loading codes and names...', end='')
        self._data = get_stock_codes_and_names()
        print('Complete!')

    def get_code_list(self) -> list:
        return list(self._data.keys())

    def get_name_list(self) -> list:
        return list(self._data.values())

    def get_name(self, code) -> str:
        if self._data is None:
            self.load_codes_and_names()

        if code in self._data:
            return self._data[code]
        return '[Unknown]'


# 获取股票的中文名称
def load_stock_code_and_names(retention_day: int = 1):
    cache_available = False
    df = pd.DataFrame(columns=['代码', '名称', '日期'])
    if os.path.exists(CODE_NAME_CACHE_PATH):
        df = pd.read_csv(CODE_NAME_CACHE_PATH, dtype={'代码': str})
        cache_date_str = df['日期'].head(1).values[0]
        cache_date = datetime.datetime.strptime(cache_date_str, '%Y-%m-%d')
        curr_date = datetime.datetime.today()
        if curr_date - cache_date < datetime.timedelta(days=retention_day):
            cache_available = True

    if not cache_available:
        try:
            df = ak.stock_info_a_code_name()
            df = df.rename(columns={'code': '代码', 'name': '名称'})

            if len(df) == 0:
                df = ak.stock_zh_a_spot()  # 这个接口容易封IP，留作备用
                df['代码'] = df['代码'].str[2:]

            df = df[['代码', '名称']]
            df = df.sort_values(by='代码')
            df['日期'] = datetime.datetime.today().strftime('%Y-%m-%d')

            df.to_csv(CODE_NAME_CACHE_PATH)
        except Exception as e:
            print('Download remote code and names failed! ', e)

    return df


def get_stock_codes_and_names() -> Dict[str, str]:
    ans = {}

    with open('./_data/mktdt00.txt', 'r', errors='replace') as r:
        lines = r.readlines()
        for line in lines:
            arr = line.split('|')
            if len(arr) > 2 and len(arr[1]) == 6:
                ans[arr[1] + '.SH'] = arr[2]

    with open('./_data/sjshq.txt', 'r', encoding='utf-8', errors='replace') as r:
        lines = r.readlines()
        for line in lines:
            arr = json.loads(line)
            ans[arr['code']] = arr['name']

    # df = ak.stock_zh_a_spot_em()
    df = load_stock_code_and_names()
    df['代码'] = df['代码'].apply(lambda x: symbol_to_code(x))
    ans.update(dict(zip(df['代码'], df['名称'])))
    return ans


# ==========
# 本地磁盘缓存
# ==========


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
def save_json(path: str, var: dict) -> None:
    with open(path, 'w') as w:
        w.write(json.dumps(var, indent=4))


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


# 所有缓存持仓天数+1，_inc_date为单日判重标记位
def all_held_inc(held_operation_lock: threading.Lock, path: str) -> bool:
    with held_operation_lock:
        held_days = load_json(path)

        today = datetime.datetime.now().strftime('%Y-%m-%d')
        inc_date_key = '_inc_date'

        try:
            if (inc_date_key not in held_days) or (held_days[inc_date_key] != today):
                held_days[inc_date_key] = today
                for code in held_days.keys():
                    if code != inc_date_key:
                        held_days[code] += 1

                save_json(path, held_days)
                return True
            else:
                return False
        except Exception as e:
            print('held days +1 failed! ', e)
            return False


# 增加新的持仓记录
def new_held(held_operation_lock: threading.Lock, path: str, codes: List[str]) -> None:
    with held_operation_lock:
        held_days = load_json(path)
        for code in codes:
            held_days[code] = 0
        save_json(path, held_days)


# 更新持仓股买入开始最高价格
def update_max_prices(
    lock: threading.Lock,
    quotes: dict,
    positions: list,
    path_max_prices: str,
    path_min_prices: str,
    path_held_days: str,
    ignore_open_day: bool = True,  # 是否忽略开仓日，从次日开始计算最高价
):
    held_days = load_json(path_held_days)

    with lock:
        max_prices = load_json(path_max_prices)
        min_prices = load_json(path_min_prices)

    max_updated = False
    min_updated = False

    for position in positions:
        code = position.stock_code
        if code in held_days:  # 只更新持仓超过一天的
            if ignore_open_day:  # 忽略开仓日的最高价
                held_day = held_days[code]
                if held_day <= 0:
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
                    if min_prices[code] < low_price:
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

    return max_prices, held_days


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


# 获取总仓位价格增幅
def get_total_asset_increase(path_assets: str, curr_date: str, curr_asset: float) -> Optional[float]:
    if os.path.exists(path_assets):
        df = pd.read_csv(path_assets)
        prev_asset = df.tail(1)['asset'].values[0]
        df.loc[len(df)] = [curr_date, curr_asset]
        df.to_csv(path_assets, index=False)
        return curr_asset - prev_asset
    else:
        df = pd.DataFrame({'date': [curr_date], 'asset': [curr_asset]})
        df.to_csv(path_assets, index=False)
        return None


# ==========
# 交易日缓存
# ==========


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
    trading_day_list = get_disk_trade_day_list_and_update_max_year()
    today = now.strftime('%Y-%m-%d')
    try:
        trading_index = list(trading_day_list).index(today)
    except ValueError:
        trading_index = np.searchsorted(trading_day_list, today) - 1

    if basic_format:
        return trading_day_list[trading_index - count].replace('-', '')
    else:
        return trading_day_list[trading_index - count]


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
            print(f'[{curr_date} is {ans} trade day in memory]')
            return ans

    # 网络缓存
    df = ak.tool_trade_date_hist_sina()
    df.to_csv(TRADE_DAY_CACHE_PATH)
    print(f'Cache trade day list {curr_year} - {int(curr_year) + 1} in {TRADE_DAY_CACHE_PATH}.')

    trade_day_list = get_disk_trade_day_list_and_update_max_year()
    if curr_year <= trade_day_cache[trade_max_year_key]:  # 未过期
        ans = curr_date in trade_day_list
        trade_day_cache[curr_date] = ans
        print(f'[{curr_date} is {ans} trade day in memory]')
        return ans

    # 实在拿不到数据默认为True
    print(f'[DO NOT KNOW {curr_date}, default to True trade day]')
    return True


def check_is_open_day(curr_date: str) -> bool:
    """
    curr_date example: '2024-12-31'
    """
    return check_is_open_day_sina(curr_date)


# ==========
# 远程数据缓存
# ==========


# 获取指数成份symbol
def get_index_constituent_symbols(index_symbol: str) -> list[str]:
    if index_symbol[:2] in ['00', '93', '89']:
        # 中证指数接口
        df = ak.index_stock_cons_csindex(symbol=index_symbol)
    else:
        # 普通指数接口：有重复不全，需要注意
        df = ak.index_stock_cons(symbol=index_symbol)

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
    df = ak.stock_zh_a_spot_em()
    df = df.sort_values('代码')
    df = df[['代码', '名称', '总市值', '流通市值']]
    df = df[(min_value < df['总市值']) & (df['总市值'] < max_value)]
    df = df[df['代码'].str.startswith(tuple(code_prefixes))]
    return [symbol_to_code(symbol) for symbol in list(df['代码'].values)]


# 获取当日可用的股票代码
def get_available_stock_codes() -> list[str]:
    df = ak.stock_info_a_code_name()
    codes = [symbol_to_code(symbol) for symbol in df['code'].values]
    return list(set(codes))


# 根据两位数前缀获取股票列表
def get_prefixes_stock_codes(prefixes: Set[str]) -> List[str]:
    """
    prefixes: 六位数的两位数前缀
    """
    df = ak.stock_info_a_code_name()
    return [symbol_to_code(symbol) for symbol in df['code'].values if symbol[:2] in prefixes]


def get_none_st_codes() -> list[str]:
    df = ak.stock_info_a_code_name()
    df = df[~df['name'].str.contains('ST')]
    df = df[~df['name'].str.contains('退市')]
    codes = [symbol_to_code(symbol) for symbol in df['code'].values]
    return list(set(codes))


# 获取流通市值，单位（元）
def get_stock_codes_and_circulation_mv() -> Dict[str, int]:
    df = ak.stock_zh_a_spot_em()
    df['代码'] = df['代码'].apply(lambda x: symbol_to_code(x))
    df = df[['代码', '流通市值']].dropna()
    return dict(zip(df['代码'], df['流通市值']))
