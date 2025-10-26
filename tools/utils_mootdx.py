import os
import logging
import io
import datetime
import json
import time
import zipfile
import numpy as np
import pandas as pd

from typing import Optional

from tdxpy.constants import SECURITY_EXCHANGE
from tdxpy.reader import TdxDailyBarReader

from tools.constants import ExitRight
from tools.utils_basic import code_to_sina_symbol, symbol_to_code, code_to_symbol
from tools.utils_cache import get_prev_trading_date_list, get_trading_date_list, get_available_stock_codes, \
                              load_pickle, save_pickle, TRADE_DAY_CACHE_PATH


DEFAULT_XDXR_CACHE_PATH = './_cache/_daily_mootdx/xdxr'

PATH_TDX_HISTORY = f'./_cache/_daily_tdxzip/history_tdxhsj.pkl'
PATH_TDX_XDXR = f'./_cache/_daily_tdxzip/xdxr.pkl'


class MootdxClientInstance:
    _instance = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MootdxClientInstance, cls).__new__(cls)
            cls.client = None  # Initialize data as None initially
        return cls._instance

    def __init__(self):
        if self.client is None:
            from mootdx.quotes import Quotes
            pd.set_option('future.no_silent_downcasting', True)
            from credentials import TDX_FOLDER
            if TDX_FOLDER is not None and len(TDX_FOLDER) > 0:
                try:
                    self.client = Quotes.factory(market='std', tdxdir=TDX_FOLDER)
                except Exception as e:
                    print('未找到本地TDX目录，使用默认TDX数据源配置：', e)
                    self.client = Quotes.factory(market='std')
            else:
                self.client = Quotes.factory(market='std')


class MootdxDailyBarReaderInstance:
    _instance = None
    reader = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MootdxDailyBarReaderInstance, cls).__new__(cls)
            cls.reader = None  # Initialize data as None initially
        return cls._instance

    def __init__(self):
        if self.reader is None:
            self.reader = MooTdxDailyBarReader()
            pd.set_option('future.no_silent_downcasting', True)


class MooTdxDailyBarReader(TdxDailyBarReader):
    """本类从mootdx复制而来，增加北交所处理，mootdx更新后需改为mootdx调用"""

    SECURITY_TYPE = [
        'SH_A_STOCK',
        'SH_B_STOCK',
        'SH_STAR_STOCK',
        'SH_INDEX',
        'SH_FUND',
        'SH_BOND',
        'SZ_A_STOCK',
        'SZ_B_STOCK',
        'SZ_INDEX',
        'SZ_FUND',
        'SZ_BOND',
        'BJ_A_STOCK',
    ]

    SECURITY_COEFFICIENT = {
        'SH_A_STOCK': [0.01, 0.01],
        'SH_B_STOCK': [0.001, 0.01],
        'SH_STAR_STOCK': [0.01, 0.01],
        'SH_INDEX': [0.01, 1.0],
        'SH_FUND': [0.001, 1.0],
        'SH_BOND': [0.001, 1.0],
        'SZ_A_STOCK': [0.01, 0.01],
        'SZ_B_STOCK': [0.01, 0.01],
        'SZ_INDEX': [0.01, 1.0],
        'SZ_FUND': [0.001, 0.01],
        'SZ_BOND': [0.001, 0.01],
        'BJ_A_STOCK': [0.01, 0.01],
    }

    def get_security_type(self, fname):

        exchange = str(fname[-12:-10]).lower()
        code_head = fname[-10:-8]

        if exchange == SECURITY_EXCHANGE[0]:
            if code_head in ['00', '30']:
                return 'SZ_A_STOCK'

            if code_head in ['20']:
                return 'SZ_B_STOCK'

            if code_head in ['39']:
                return 'SZ_INDEX'

            if code_head in ['15', '16', '18']:
                return 'SZ_FUND'

            if code_head in ['10', '11', '12', '13', '14']:
                return 'SZ_BOND'

            return 'SZ_OTHER'

        if exchange == SECURITY_EXCHANGE[1]:
            if code_head in ['60']:
                return 'SH_A_STOCK'

            if code_head in ['90']:
                return 'SH_B_STOCK'

            if code_head in ['68']:
                return 'SH_STAR_STOCK'

            if code_head in ['00', '88', '99']:
                return 'SH_INDEX'

            if code_head in ['50', '51', '58']:
                return 'SH_FUND'

            # if code_head in ['01', '10', '11', '12', '13', '14']:
            if code_head in ['01', '02', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']:
                return 'SH_BOND'

            return 'SH_OTHER'

        if exchange == SECURITY_EXCHANGE[2]:
            if code_head in ['43', '82', '83', '87', '88', '92']:
                return 'BJ_A_STOCK'

        logging.error('Unknown security exchange !\n')
        raise NotImplementedError



def make_qfq(data, xdxr, fq_type="01"):
    """使用数据库数据进行复权"""

    # 过滤其他，只留除权信息
    xdxr = xdxr.query("category==1")
    # data = data.assign(if_trade=1)

    if len(xdxr) > 0:
        # 有除权信息, 合并原数据 + 除权数据
        # data = pd.concat([data, xdxr.loc[data.index[0]:data.index[-1], ['category']]], axis=1)
        # data['if_trade'].fillna(value=0, inplace=True)

        data = data.ffill()
        # present       bonus       price       rationed
        # songzhuangu   fenhong     peigujia    peigu
        data = pd.concat(
            [data, xdxr.loc[data.index[0]:data.index[-1], ["fenhong", "peigu", "peigujia", "songzhuangu"]]],
            axis=1,
        )
    else:
        # 没有除权信息
        data = pd.concat([data, xdxr.loc[:, ["fenhong", "peigu", "peigujia", "songzhuangu"]]], axis=1)

    # 清理数据
    data = data.fillna(0)

    if fq_type == "01":
        data["preclose"] = (
                (data["close"].shift(1) * 10 - data["fenhong"] + data["peigu"] * data["peigujia"]) /
                (10 + data["peigu"] + data["songzhuangu"])
        )
        # 生成 adj 复权因子
        data["adj"] = (data["preclose"].shift(-1) / data["close"]).fillna(1)[::-1].cumprod()
    else:
        # 生成 preclose 关键位置
        data["preclose"] = (
                (data["close"].shift(1) * 10 - data["fenhong"] + data["peigu"] * data["peigujia"]) /
                (10 + data["peigu"] + data["songzhuangu"])
        )
        # 生成 adj 复权因子
        data["adj"] = (data["close"] / data["preclose"].shift(-1)).cumprod().shift(1).fillna(1)

    # 计算复权价格
    for field in data.columns.values:
        if field in ("open", "close", "high", "low", "preclose"):
            data[field] = data[field] * data["adj"]

    data[["open", "close", "high", "low", "preclose"]] = data[["open", "close", "high", "low", "preclose"]].round(3)

    # 清理数据, 返回结果
    return data.query("open != 0").drop([
        "fenhong", "peigu", "peigujia", "songzhuangu"
    ], axis=1)


def make_hfq(bfq_data, xdxr_data):
    """使用数据库数据进行复权"""
    info = xdxr_data.query('category==1')
    bfq_data = bfq_data.assign(if_trade=1)

    if len(info) > 0:
        # 合并数据
        data = pd.concat([bfq_data, info.loc[bfq_data.index[0]:bfq_data.index[-1], ['category']]], axis=1)
        data['if_trade'] = data['if_trade'].fillna(value=0)

        data = data.ffill()
        data = pd.concat([
            data,
            info.loc[bfq_data.index[0]:bfq_data.index[-1], ['fenhong', 'peigu', 'peigujia', 'songzhuangu']]
        ], axis=1)
    else:
        data = pd.concat([bfq_data, info.loc[:, ['category', 'fenhong', 'peigu', 'peigujia', 'songzhuangu']]], axis=1)

    data = data.fillna(0)

    # 生成 preclose 关键位置
    data['preclose'] = (data['close'].shift(1) * 10 - data['fenhong'] + data['peigu'] * data['peigujia']) / \
                       (10 + data['peigu'] + data['songzhuangu'])
    data['adj'] = (data['close'] / data['preclose'].shift(-1)).cumprod().shift(1).fillna(1)

    # 计算复权价格
    for field in data.columns.values:
        if field in ('open', 'close', 'high', 'low', 'preclose'):
            data[field] = data[field] * data['adj']

    # data['open'] = data['open'] * data['adj']
    # data['high'] = data['high'] * data['adj']
    # data['low'] = data['low'] * data['adj']
    # data['close'] = data['close'] * data['adj']
    # data['preclose'] = data['preclose'] * data['adj']

    # 不计算 交易量
    # data['volume'] = data['volume'] / data['adj'] if 'volume' in data.columns else data['vol'] / data['adj']

    try:
        data['high_limit'] = data['high_limit'] * data['adj']
        data['low_limit'] = data['high_limit'] * data['adj']
    except Exception as e:
        print('xdxr error! ', e)

    data[["open", "close", "high", "low", "preclose"]] = data[["open", "close", "high", "low", "preclose"]].round(3)

    return data.query('if_trade==1 and open != 0').drop([
        'fenhong', 'peigu', 'peigujia', 'songzhuangu', 'if_trade', 'category'
    ], axis=1)


def _get_offset_start(csv_path: str, start_date_str: str, end_date_str: str) -> tuple[int, int]:
    """
    计算两个日期区间的交易日数（含首尾），及end到今天的交易日数（不含end）

    参数：
    csv_path: str - 交易日CSV文件路径（需包含trade_date列）
    start_date_str: str - 起始日期（格式：20250101）
    end_date_str: str - 结束日期（格式：20250101）

    返回：
    tuple - (start到end的交易日数, end到今天的交易日数)
    """
    # 1. 读取并预处理交易日数据（确保排序、去重）
    df = pd.read_csv(csv_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date  # 仅保留日期部分（避免时间干扰）
    trade_dates = df["trade_date"].drop_duplicates().sort_values().reset_index(drop=True)
    if trade_dates.empty:
        return 0, 0  # 无交易日数据时返回0

    # 2. 工具函数：调整日期为目标方向的最近交易日
    def adjust_date(target_date, direction):
        """
        direction: 'next'（start专用：取>=target的最近交易日）
                  'prev'（end/今天专用：取<=target的最近交易日）
        """
        # 超出交易日范围时返回无效标记
        if target_date < trade_dates.min():
            return trade_dates.min() if direction == 'next' else None
        if target_date > trade_dates.max():
            return None if direction == 'next' else trade_dates.max()

        # 二分查找快速定位索引
        if direction == 'next':
            idx = trade_dates.searchsorted(target_date, side="left")
            return trade_dates.iloc[idx]
        else:
            idx = trade_dates.searchsorted(target_date, side="right") - 1
            return trade_dates.iloc[idx] if idx >= 0 else None

    # 3. 解析输入日期（转为date类型，与trade_dates格式统一）
    now = datetime.datetime.now()
    curr_date = now.date()  # 今天的日期（仅日期部分）
    try:
        today_str = curr_date.strftime("%Y%m%d")
        start_date = datetime.datetime.strptime(start_date_str, "%Y%m%d").date()
        end_date = datetime.datetime.strptime(min(end_date_str, today_str), "%Y%m%d").date()
    except ValueError:
        return 0, 0  # 日期格式错误返回0

    # ---------------------- 4. 计算：start到end的交易日数（含首尾） ----------------------
    adjusted_start = adjust_date(start_date, direction="next")  # start非交易日则取后一个
    adjusted_end = adjust_date(end_date, direction="prev")      # end非交易日则取前一个

    days_between = 0
    if adjusted_start is not None and adjusted_end is not None and adjusted_start <= adjusted_end:
        # 索引差 +1 = 含首尾的总天数（如16日索引0、17日1、18日2：2-0+1=3）
        idx_start = trade_dates[trade_dates == adjusted_start].index[0]
        idx_end = trade_dates[trade_dates == adjusted_end].index[0]
        days_between = idx_end - idx_start + 1

    # ---------------------- 5. 计算：end到今天的交易日数（不含end） ----------------------

    adjusted_today = adjust_date(curr_date, direction="prev")  # 今天非交易日则取前一个

    days_from_end_to_today = 0
    if adjusted_end is not None and adjusted_today is not None and adjusted_end < adjusted_today:
        # 索引差 = 不含end的天数（如18日索引2、19日3：3-2=1）
        idx_adjusted_end = trade_dates[trade_dates == adjusted_end].index[0]
        idx_today = trade_dates[trade_dates == adjusted_today].index[0]
        days_from_end_to_today = idx_today - idx_adjusted_end

    # 早上有当日的daily的K线之前要少向前推一天
    if trade_dates.isin([curr_date]).any() and now.time() < datetime.time(9, 30):
        if days_from_end_to_today > 0:
            days_between += 1
            days_from_end_to_today -= 1
    else:
        days_between += 1

    return days_between, days_from_end_to_today


def _get_bars_with_offset(client, symbol, total_offset, start=0):
    all_dfs = []
    remaining = total_offset
    current_start = start
    batch_size = 800  # 每次最大获取数量
    datetime_col = 'datetime'  # 假设时间列名为'datetime'，根据实际情况调整

    while remaining > 0:
        fetch_count = min(remaining, batch_size)

        try:
            df = client.bars(
                symbol=symbol,
                frequency='day',
                offset=fetch_count,
                start=current_start,
            )

            if df is None or df.empty:
                break  # 没有更多数据

            # 检查是否存在时间列，避免后续排序出错
            if datetime_col not in df.columns:
                print(f"Error: 数据中缺少'{datetime_col}'列，无法排序")
                return None

            all_dfs.append(df)
            remaining -= fetch_count
            current_start += fetch_count  # 移动到下一批的起始位置

        except Exception as e:
            print(f'mootdx get daily {symbol} error: ', e)
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True).iloc[:total_offset]
                return combined.sort_values(by=datetime_col, ascending=True).reset_index(drop=True)
            return None

    if not all_dfs:
        return None

    # 合并所有数据并截取总数量
    combined_df = pd.concat(all_dfs, ignore_index=True).iloc[:total_offset]

    # 按datetime列从大到小排序（最新时间在前）
    # 若时间列已转为datetime类型，排序会更准确
    combined_df[datetime_col] = pd.to_datetime(combined_df[datetime_col])  # 确保是datetime类型
    combined_df = combined_df.sort_values(by=datetime_col, ascending=True).reset_index(drop=True)

    return combined_df


def _get_xdxr(symbol: str, cache_dir: str = DEFAULT_XDXR_CACHE_PATH, expire_hours: int = 12) -> Optional[pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}.csv")  # 缓存文件名：股票代码.csv
    expire_seconds = expire_hours * 3600

    if os.path.exists(cache_file):
        file_mtime = os.path.getmtime(cache_file)
        time_diff = time.time() - file_mtime

        if time_diff <= expire_seconds:
            return pd.read_csv(cache_file)

    try:
        client = MootdxClientInstance().client
        xdxr_data = client.xdxr(symbol=symbol)
        if xdxr_data is not None and isinstance(xdxr_data, pd.DataFrame):
            xdxr_data.to_csv(cache_file, index=False)

        return xdxr_data
    except Exception as e:
        print(f' mootdx get xdxr {symbol} error: ', e)
        return None


# 获取 mootdx 的历史日线
# 使用 mootdx 数据源记得 pip install mootdx
def get_mootdx_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    offset, start = _get_offset_start(TRADE_DAY_CACHE_PATH, start_date, end_date)
    symbol = code_to_symbol(code)

    client = MootdxClientInstance().client
    try:
        df = _get_bars_with_offset(client, symbol, offset, start)
        # TODO_List: 对于有些期间停牌过的票，发现时间对不上这里要校正，优先级不高因为只会多不会少
    except Exception as e:
        print(f' mootdx get daily {code} error: ', e)
        return None

    if adjust != ExitRight.BFQ:
            xdxr = _get_xdxr(symbol=symbol)
            if xdxr is not None and isinstance(xdxr, pd.DataFrame) and len(xdxr) > 0:
                xdxr['date_str'] = xdxr['year'].astype(str) + \
                    '-' + xdxr['month'].astype(str).str.zfill(2) + \
                    '-' + xdxr['day'].astype(str).str.zfill(2)
                xdxr['datetime'] = pd.to_datetime(xdxr['date_str'] + ' 15:00:00')
                xdxr = xdxr.set_index('datetime')
                xdxr_info = xdxr.loc[xdxr['category'] == 1]

                # 默认除权日当天之前的数据一样进行处理
                is_appended = False
                now = datetime.datetime.now()
                if not xdxr_info.empty and xdxr_info.index[-1].date() <= now.date():
                    curr_date = now.strftime("%Y-%m-%d") + ' 15:00:00'
                    last_row = df.iloc[-1].copy()
                    last_row['datetime'] = curr_date
                    df.loc[len(df)] = last_row
                    df.index = pd.to_datetime(df['datetime'].astype(str), errors="coerce")
                    is_appended  = True

                try:
                    if adjust == ExitRight.QFQ:
                        df = make_qfq(df, xdxr)
                    elif adjust == ExitRight.HFQ:
                        df = make_hfq(df, xdxr)
                except Exception as e:
                    print(f' mootdx make fq {code} error: ', e)

                if is_appended:
                    df = df[:-1]

    if df is not None and len(df) > 0 and isinstance(df, pd.DataFrame) and 'datetime' in df.columns:
        try:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()  # 然后删除所有包含 NaN 的行
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['datetime'] = df['datetime'].dt.date.astype(str).str.replace('-', '').astype(int)
            df = pd.concat([df['datetime'], df.drop('datetime', axis=1)], axis=1)

            df = df.drop(columns='volume')
            df = df.rename(columns={'vol': 'volume'})
            df['volume'] = df['volume'].astype(int)
            df = df.reset_index(drop=True)

            if columns is not None:
                return df[columns]
            return df
        except Exception as e:
            print(f' handle format {code} error: ', e)
            return None
    return None


# ================================================
# TDX_ZIP
# ================================================


def _get_xdxr_sina(code: str, adjust: ExitRight, factor_name: str = None) -> pd.DataFrame:
    xdxr = MootdxClientInstance().client.xdxr(symbol=code_to_symbol(code))
    if xdxr is not None and len(xdxr) > 0:
        xdxr['date_str'] = xdxr['year'].astype(str) + \
                           '-' + xdxr['month'].astype(str).str.zfill(2) + \
                           '-' + xdxr['day'].astype(str).str.zfill(2)
        xdxr['datetime'] = pd.to_datetime(xdxr['date_str'])
        xdxr = xdxr.set_index('datetime')

        fq = _fetch_xdxr_factor(code, adjust, factor_name)
        xdxr = xdxr.join(fq[1:], how='outer')
    return xdxr


def _fq_factor(code: str, method: str, ) -> pd.DataFrame:
    '''
    本函数从mootdx 剥离出来，当前mootdx不能处理北交所
    未来可考虑直接调用mootdx 的factor.fq_factor
    参数 sina_symbol 为sina格式 ，需使用code_to_syna_symbol转换
    '''
    symbol = code_to_sina_symbol(code)
    #symbol = symbol.replace('sh', '').replace('sz', '').replace('bj', '')
    #market = _get_stock_market(symbol, string=True)
    #symbol = f'{market}{symbol}'
    #cache_file = get_config_path(f'caches/factor/{symbol}_{method}.plk') #弃用

    #@file_cache(filepath=cache_file, refresh_time=3600 * 24) #弃用
    def _factor(symbol: str, method: str, ) -> pd.DataFrame:
        import httpx
        headers = {
            "referer": f'https://finance.sina.com.cn/realstock/company/{symbol}/nc.shtml',
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0",
        }
        try:
            url = "https://finance.sina.com.cn/realstock/company/{}/{}.js"
            rsp = httpx.get(url.format(symbol, method), headers=headers)
            res = pd.DataFrame(eval(rsp.text.split("=")[1].split("\n")[0])["data"])
        except (SyntaxError, httpx.ConnectError) as ex:
            logging.error(ex)
            return pd.DataFrame(None)

        if res.shape[0] == 0:
            raise ValueError(f"sina {method} factor not available")

        res.columns = ["date", "factor"]
        res.date = pd.to_datetime(res.date)

        res.set_index("date", inplace=True)
        time.sleep(0.3)
        return res

    return _factor(symbol, method)

def _fetch_xdxr_factor(code, adjust, factor_name=None) -> pd.DataFrame:
    if factor_name is None:
        factor_name = f'{adjust}_factor'
    '''
    symbol = code_to_symbol(code)
    # 直接使用factor.fq_factor 需要清理以下缓存文件避免错误，使用本文件内_fq_factor 不需要
    market = _get_stock_market(symbol, string=True)
    cache_file = factor.get_config_path(f'caches/factor/{market}{symbol}_{adjust}.plk')
    if os.path.isfile(cache_file):
        os.remove(cache_file)
    '''
    fq = _fq_factor(code, adjust)
    fq.rename(columns={'factor': factor_name}, inplace=True)
    fq.sort_index(ascending=True, inplace=True)
    return fq


def _factor_reversion(method: str = 'qfq', raw: pd.DataFrame = None, adj_factor: pd.DataFrame = None) -> pd.DataFrame:
    if adj_factor is not None and not adj_factor.empty:
        raw.index = pd.to_datetime(raw.index)
        adj_factor.index = pd.to_datetime(adj_factor.index)

        # 按日期升序排列复权因子
        adj_factor = adj_factor.sort_index(ascending=True)
        raw = raw.sort_index(ascending=True)
        # 获取原始数据期间的复权因子
        # 使用最近的可用的复权因子（向后填充）
        adj_factor = adj_factor.reindex(raw.index, method='ffill')

        data = pd.concat([raw, adj_factor.loc[raw.index[0]: raw.index[-1], ['factor']]], axis=1)
        data.factor = data.factor.fillna(1.0, axis=0)
        data.factor = data.factor.astype(float)
        if method == 'qfq':
            for col in ['open', 'high', 'low', 'close', ]:
                data[col] = data[col] / data['factor']
        elif method == 'hfq':
            for col in ['open', 'high', 'low', 'close', ]:
                data[col] = data[col] * data['factor']
        return data
    raw['factor'] = 1.0
    return raw


def update_tdx_hsjday(TDXDIR: str, isExtract = True, cachefile = None) -> bool:
    """
    从通达信网站下载沪深京日线数据，并解压到通达信目录下

    Args:
        TDXDIR: 通达信所在目录例如：C:/new_tdx

    Returns:
        bool: 成功返回True，失败返回False
    """

    try:
        print(f"开始下载通达信沪深京日线数据文件。")
        start = datetime.datetime.now().timestamp()
        buffer = download_tdx_hsjday()
        end = datetime.datetime.now().timestamp()

        if not buffer:
            return False

        buffer.seek(0, io.SEEK_END)
        response_length = buffer.tell()
        file_size = response_length / 1024 / 1024
        print(f'下载耗时{end-start:.2f}秒, 速度：{file_size/(end-start):.2f} MB/s。')
        if isExtract:
            vip_doc_dir = os.path.join(TDXDIR, 'vipdoc')
            os.makedirs(vip_doc_dir, exist_ok=True)
            end = datetime.datetime.now().timestamp()

            file_num = 0
            print(f"已下载，文件大小：{file_size:.2f}MB。开始解压文件到: {TDXDIR} 。")
            with zipfile.ZipFile(buffer, 'r') as zip_ref:
                zip_ref.extractall(vip_doc_dir)
                file_num = len(zip_ref.infolist())
            end2 = datetime.datetime.now().timestamp()
            print(f'解压耗时{end2-end:.2f}秒，解压文件{file_num}个。')

            print("下载通达信沪深京日线数据并解压完成。")
        if cachefile:
            with open(cachefile, 'wb') as file:
                file.write(buffer.getvalue())
            print(f"文件已写入{cachefile}。")

        buffer.close()
        del buffer
        return True
    except zipfile.BadZipFile as e:
        error_msg = f"解压文件失败，文件可能已损坏: {str(e)}"
        print(error_msg)
        return False
    except PermissionError as e:
        error_msg = f"权限错误，无法写入目录: {str(e)}"
        print(error_msg)
        return False
    except Exception as e:
        import traceback
        error_msg = f"未知错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False


def _process_tdx_zip_to_datas(group_codes, zip_ref, cache_xdxr, day_count, adjust):
    """处理tdx zip文件，未来需改造多线程"""
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    result = {}
    factor_name = f'{adjust}_factor'
    default_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adj']
    barreader = MootdxDailyBarReaderInstance().reader

    for code in group_codes:
        arr = code.split('.')
        assert len(arr) == 2, 'code不符合格式'
        symbol = arr[0]
        market = arr[1].lower()
        filename = f'{market}/lday/{market}{symbol}.day'

        try:
            member = zip_ref.getinfo(filename)
        except KeyError:
            # 文件不存在
            result[code] = None, code, 'maybe is new stock'
            continue
        df = None
        try:
            with zip_ref.open(member) as source:
                # 以下代码来自 TdxDailyBarReader.get_df_by_file()
                security_type = barreader.get_security_type(filename)

                if security_type not in barreader.SECURITY_TYPE:
                    result[code] = None, code, 'security_type'
                    continue

                coefficient = barreader.SECURITY_COEFFICIENT[security_type]
                content = barreader.unpack_records("<IIIIIfII", source.read())
                data = [barreader._df_convert(row_, coefficient) for row_ in content]

                df = pd.DataFrame(data=data[-day_count:], columns=["datetime", "open", "high", "low", "close", "amount", "volume"])
                df.index = pd.to_datetime(df['datetime'], errors="coerce")
        except Exception as e:
            print('x', end='')
            result[code] = None, code, f'extract zip error: str{e}'
            continue

        try:
            xdxr = cache_xdxr.get(code, None)
            if xdxr is None:
                xdxr = _get_xdxr_sina(code, adjust, factor_name)
                if xdxr is not None and len(xdxr) > 0:
                    cache_xdxr[code] = xdxr

            if xdxr is not None and len(xdxr) > 0 and adjust and adjust in [ExitRight.QFQ, ExitRight.HFQ] :
                try:
                    if symbol == '689009':
                        raise Exception('CDR公司')
                    if factor_name not in xdxr.columns: # 没有复权因子数据需要更新
                        fq = _fetch_xdxr_factor(code, adjust, factor_name)
                        xdxr = xdxr.join(fq[1:], how='outer')
                        cache_xdxr[code] = xdxr
                    else:
                        fq = xdxr.loc[xdxr['category'] == 1, [factor_name]]
                        fq.index.name = 'date'
                        fq.rename(columns={factor_name:'factor'}, inplace=True)
                    xdxr_info = xdxr.loc[xdxr['category'] == 1]
                    if not xdxr_info.empty and xdxr_info.index[-1].date() == now.date():
                        if fq.index[-1].date() != now.date() or pd.isna(xdxr_info.iloc[-1][factor_name]) == True:
                            raise Exception('缺少今日除权因子数据')
                    df = _factor_reversion(adjust, df, fq)
                    df.rename(columns={'factor': 'adj'}, inplace=True)
                except Exception:
                    is_append = False
                    xdxr_info = xdxr.loc[xdxr['category'] == 1]
                    if not xdxr_info.empty and xdxr_info.index[-1].date() <= now.date():
                        last_row = df.iloc[-1].copy()
                        last_row['datetime'] = curr_date
                        df.loc[len(df)] = last_row
                        df.index = pd.to_datetime(df['datetime'].astype(str), errors="coerce")
                        is_append = True

                    if adjust == ExitRight.QFQ:
                        df = make_qfq(df, xdxr)
                    elif adjust == ExitRight.HFQ:
                        df = make_hfq(df, xdxr)
                    if is_append:
                        df = df[:-1]
        except Exception as e:
            result[code] = df, code, f'xdxr error: {e}'
            continue

        if df is not None and len(df) > 0 and isinstance(df, pd.DataFrame) and 'datetime' in df.columns:
            try:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)  # 然后删除所有包含 NaN 的行
                df['volume'] = df['volume'].astype(int)
                df['datetime'] = df['datetime'].str.replace('-', '').astype(int)
                df = df.reset_index(drop=True)
                if 'adj' not in df.columns:
                    df['adj'] = 1.0
                result[code] = df[default_columns], code, None
            except Exception as e:
                print(f'x[{e}]', end='')
                result[code] = None, code, 'format error: str{e}'
                continue
        else:
            result[code] = None, code, 'no data'
            continue
    return result


# 检查xdxr缓存，建议定时调度运行
def check_xdxr_cache(adjust=ExitRight.QFQ, force_refresh_updated_date: bool = False) -> None:
    """
    检查缓存的通达信除权除息数据，cache_xdxr数据格式为stockcode:pd.DataFrame，以及updatedtime: Datetime
    如果发现有除权除息日大于updatetime 且小于当前日期，就删除cache条目
    """
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    try:
        cache_xdxr = load_pickle(PATH_TDX_XDXR)
        if cache_xdxr is None or not isinstance(cache_xdxr, dict):
            logging.warning('未能加载除权除息缓存文件')
            print('未能加载除权除息缓存文件')
            return

        if cache_xdxr.get('updatedtime', None) is None or not isinstance(cache_xdxr['updatedtime'], datetime.datetime):
            date_list = get_prev_trading_date_list(curr_date, 20)
        else:
            updated_date = cache_xdxr['updatedtime'].strftime('%Y-%m-%d')
            date_list = get_trading_date_list(updated_date, curr_date)
            if not force_refresh_updated_date:
                date_list = date_list[1:]

        factor_name = f'{adjust}_factor'
        removed_xdxr_codes = set()

        for cdate in date_list:
            try:
                xcdf, divicount = get_dividend_code_from_baidu(cdate)  #该接口返回的df中code实际上是symbol，注意转换
            except Exception as e:
                logging.warning(f'从百度获取 {cdate} 除权股票列表数据出现问题：{str(e)}')
                print(f'从百度获取 {cdate} 除权股票列表数据出现问题：{str(e)}')
                continue
            if divicount != len(xcdf):
                logging.warning(f'从百度获取 {cdate} 除权股票列表数据不完整。期望获得{divicount}，实际获得{len(xcdf)}')
                print(f'从百度获取 {cdate} 除权股票列表数据不完整。期望获得{divicount}，实际获得{len(xcdf)}')

            for row in xcdf.itertuples():
                symbol = row.code
                code = symbol_to_code(symbol)
                xdxr = cache_xdxr.get(code, None)
                if xdxr is None: # 没有该股票的除权除息信息，需重新获取
                    removed_xdxr_codes.add(symbol)
                    continue

                curr_xc = xdxr.loc[xdxr['category'] == 1]
                if (
                    not curr_xc.empty
                    and factor_name in curr_xc.columns
                    and curr_xc.iloc[-1]["date_str"] == cdate
                    and (
                        pd.isna(curr_xc.iloc[-1][factor_name])
                        or abs(float(curr_xc.iloc[-1][factor_name]) - 1.0) <= 0.001
                    )
                ):
                    continue
                removed_xdxr_codes.add(symbol)

        if len(removed_xdxr_codes) > 0:
            print(f'{len(removed_xdxr_codes)}只股票需要更新复权因子。')
            success_count = 0
            for symbol in removed_xdxr_codes:
                code = symbol_to_code(symbol)
                try:
                    xdxr = _get_xdxr_sina(code, adjust, factor_name)
                    if xdxr is not None and not xdxr.empty:
                        curr_xc = xdxr.loc[xdxr['category'] == 1]
                        if not curr_xc.empty and factor_name in curr_xc.columns and float(curr_xc.iloc[-1][factor_name]) == 1.0 :
                            cache_xdxr[code] = xdxr
                            success_count += 1
                            continue
                    logging.error(f'更新{code}除权除息数据有异常，无有效数据。')
                    print(f'更新{code}除权除息数据有异常，无有效数据。')
                except Exception as e:
                    logging.error(f'更新{code}除权除息数据失败，错误:{str(e)}')
                    print(f'更新{code}除权除息数据失败，错误:{str(e)}')
            if len(removed_xdxr_codes) == success_count: # 成功更新才更新时间，让第二天再次更新
                cache_xdxr['updatedtime'] = datetime.datetime.now()
            save_pickle(PATH_TDX_XDXR, cache_xdxr)
            print(f'成功更新{success_count}只股票复权因子。')
    except Exception as e: #异常不要紧，不要因为异常影响实际运行
        logging.error(f'处理除权除息数据出现问题：{str(e)}')


def _pycurl_request(url, headers=None):
    import pycurl
    try:
        buffer = io.BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(pycurl.TIMEOUT, 30)
        c.setopt(pycurl.FOLLOWLOCATION, 1)
        c.setopt(pycurl.MAXREDIRS, 5)
        if headers is not None and len(headers)>0:
            if isinstance(headers, dict):
                headers = [f'{key}: {value}' for key, value in headers.items()]
            c.setopt(pycurl.HTTPHEADER, headers)
        #c.setopt(c.VERBOSE, True)
        c.setopt(pycurl.SSL_VERIFYPEER, 0)
        c.setopt(pycurl.SSL_VERIFYHOST, 0)

        c.setopt(c.WRITEDATA, buffer)
        c.perform()
        response_code = c.getinfo(c.RESPONSE_CODE)
        c.close()
        if response_code != 200:  #返回状态不对，或者返回文件太小
            print(f'下载文件{url}失败')
            buffer.close()
            del buffer
            return False, None
        response_context = buffer.getvalue()
        buffer.close()
        return response_code, response_context
    except Exception as e:
        print(f'下载文件失败 {str(e)}')
        return False, None


def _get_stock_dividend(headers: dict, start_date: str, page: int = 0) -> pd.DataFrame:
    divi_df, divi_count, has_more = pd.DataFrame(), 0, False
    if len(start_date) == 8:  # 20241107 -> 2024-11-07
        start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
    try:
        url = f"https://finance.pae.baidu.com/sapi/v1/financecalendar?start_date={start_date}&end_date={start_date}&market=ab&pn={page}&rn=100&cate=notify_divide&finClientType=pc"
        headers.update({'Accept': 'application/vnd.finance-web.v1+json'})
        status_code, content = _pycurl_request(url, headers=headers)
        if status_code != 200:
            raise Exception(f'百度股市通接口异常，返回HTTP 状态码为{status_code}')
            # return divi_df, divi_count, has_more
        data_json = json.loads(content)
        if data_json.get('status', 0)!=0 or data_json.get('ResultCode', -1) != 0: #出错了
            raise Exception(f'百度股市通接口异常，返回HTTP 状态码为{data_json.get("status", 0)} {data_json.get("ResultCode", -1)}')
            # return divi_df, divi_count, has_more

        for item in data_json['Result']['calendarInfo']:
            divi_data = item["list"]
            divi_count += item['total']
            has_more = bool(item['hasMore'])
            if len(divi_data) > 0:
                divi_df = pd.concat([divi_df, pd.DataFrame(divi_data)], ignore_index=True)

        return divi_df[["code","name","date"]], divi_count, has_more #
    except Exception as e:
        raise Exception(f'百度股市通接口异常：{str(e)}')
        # return divi_df, divi_count, has_more


def get_dividend_code_from_baidu(date: str = "20241107") -> (pd.DataFrame, int):
    """
    #该接口返回的df中code实际上是symbol，注意转换
    from AKShare
    百度股市通-交易提醒-分红派息
    https://gushitong.baidu.com/calendar
    :param date: 查询日期
    :type date: str
    :return: 交易提醒-分红派息
    :rtype: pandas.DataFrame, int
    """

    #  函数主体
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
        'Referer': 'https://gushitong.baidu.com/',
        'Origin': 'https://gushitong.baidu.com',
    }

    # 不检查 cookie 就先省略掉
    # url = 'https://gushitong.baidu.com/calendar'
    # status_code, content = _pycurl_request(url, headers=headers)

    page = 0
    res_df = []
    while True:
        df, divi_count, has_more = _get_stock_dividend(headers, date, page)
        if len(res_df) == 0 and has_more == False:  # 第一次调用的时候如果没有更多分页直接返回
            return df, divi_count
        if not df.empty:
            res_df.append(df)

        if not has_more:
            return pd.concat(res_df, ignore_index=True), divi_count
        page += 1
        time.sleep(0.5)

    # TODO miss return?


# 从通达信网站下载日线文件，每日盘前或盘后16:00之后下载，建议盘前
def download_tdx_hsjday() -> io.BytesIO|bool:
    import pycurl
    url = "https://data.tdx.com.cn/vipdoc/hsjday.zip"
    try:
        buffer = io.BytesIO()
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(pycurl.HTTPHEADER, ['Host: data.tdx.com.cn', 'User-Agent: curl/8.14.1', 'Accept: */*'])
        c.setopt(c.WRITEDATA, buffer)
        c.setopt(pycurl.SSL_VERIFYPEER, 0)
        c.setopt(pycurl.SSL_VERIFYHOST, 0)
        c.perform()
        response_code = c.getinfo(c.RESPONSE_CODE)
        c.close()

        response_length = buffer.tell()
        if response_code != 200 or response_length <= 102400:  #返回状态不对，或者返回文件太小
            print(f'下载文件{url}失败')
            buffer.close()
            del buffer
            return False
        return buffer
    except Exception as e:
        print(f'下载文件失败', e)
        return False


def get_tdxzip_history(adjust: ExitRight = ExitRight.QFQ, day_count: int = 550) -> dict:
    """
    直接从通达信网站下载日线文件加载到cache_history，且完成前复权计算
    TDX hsjday.zip 缓存在./_cache/_daily_tdxzip/ 目录下，除权除息缓存文件也在xdxr.pkl目录下
    """
    cache_history = {}
    cache_history_path = PATH_TDX_HISTORY
    cachepath = os.path.dirname(cache_history_path)
    os.makedirs(cachepath, exist_ok=True)
    if os.path.isfile(cache_history_path) and os.path.getmtime(cache_history_path) > time.mktime(datetime.date.today().timetuple()): #当天文件才加载
        cache_history = load_pickle(cache_history_path)
        return cache_history

    code_list = get_available_stock_codes()

    print(f'[HISTORY] Downloading {len(code_list)} gap codes data of {day_count} days.')
    tdx_hsjday_file = f'{cachepath}/hsjday.zip'
    cache_xdxr_file = PATH_TDX_XDXR

    buffer = None
    try:
        if not os.path.exists(tdx_hsjday_file) or os.path.getmtime(tdx_hsjday_file) < time.mktime(datetime.date.today().timetuple()):
            start = time.time()
            buffer = download_tdx_hsjday()
            end = time.time()
            if buffer == False:
                return cache_history
            buffer.seek(0, io.SEEK_END)
            response_length = buffer.tell()
            file_size = response_length / 1024 / 1024
            print(f'[HISTORY] 下载通达信日线文件耗时{end-start:.2f}秒, 速度：{file_size/(end-start):.2f} MB/s。')
            with open(tdx_hsjday_file, 'wb') as file:
                file.write(buffer.getvalue())
                print(f"[HISTORY] 通达信日线文件已写入 {tdx_hsjday_file}。")
        else:
            with open(tdx_hsjday_file, "rb") as fh:
                buffer = io.BytesIO(fh.read())

        cache_xdxr = load_pickle(cache_xdxr_file)
        if cache_xdxr is None or not isinstance(cache_xdxr, dict):
            print('[HISTORY] 未能加载除权除息缓存文件，将重新生成')
            cache_xdxr = {}

        # 初始化计数器和列表（多线程下需考虑线程安全）
        downloaded_count = 0
        download_failure = []

        start = time.time()

        with zipfile.ZipFile(buffer, 'r') as zip_ref:
            result_dict = _process_tdx_zip_to_datas(code_list, zip_ref, cache_xdxr, day_count, adjust)
            for code in result_dict:
                try:
                    df, _code, error_type = result_dict[code]
                    if df is not None:
                        cache_history[code] = df
                        downloaded_count += 1
                        if str(error_type).startswith('xdxr error'):
                            download_failure.append(code)
                            print(f'{code}: {error_type}')
                    else:
                        download_failure.append(code)
                        print(f'{code}: {error_type}')
                except Exception as e:
                    print('下载tdx数据包失败：', e)
                    download_failure.append(code)

        error_count = len(download_failure)

        end = time.time()
        buffer.close()
        print(f'[HISTORY] Download finished with {downloaded_count} code, Elapsed time: {end-start:.2f}s, {error_count} errors and failed with {len(download_failure)} fails: {download_failure}')
        if len(cache_xdxr) > 0:
            save_pickle(PATH_TDX_XDXR, cache_xdxr)
        del buffer
        save_pickle(PATH_TDX_HISTORY, cache_history)
        return cache_history
    except Exception as ex:
        print(f'[HISTORY] get tdx hsjday date error :', ex)
        return cache_history
