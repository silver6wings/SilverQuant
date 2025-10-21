import os
import csv
import time
import datetime
import io
import zipfile

import requests
import numpy as np
import pandas as pd
from typing import Optional

from tools.constants import DataSource, ExitRight
from tools.utils_basic import *
from tools.utils_cache import TRADE_DAY_CACHE_PATH, get_available_stock_codes, load_pickle, save_pickle

from tools.utils_mootdx import MootdxClientInstance, get_offset_start, execute_fq, get_xdxr, \
        download_tdx_hsjday, _process_tdx_zip_to_datas, PATH_TDX_HISTORY, PATH_TDX_XDXR


def set_tdx_zxg_code(data: list[str], file_name: str = None) -> None:
    if file_name is None:
        try:
            from credentials import TDX_FOLDER
            file_name = TDX_FOLDER + r'\T0002\blocknew\ZXG.blk'  # 自选股文件
        except Exception as exception:
            print('未找到tdx配置路径，放弃写入自选股', exception)
            return

    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([code_to_tdxsymbol(item)])
    print(f'已成功将数据写入自选股文件：{file_name}')


def get_tdx_zxg_code(file_name: str = None) -> list[str]:
    if file_name is None:
        try:
            from credentials import TDX_FOLDER
            file_name = TDX_FOLDER + r'\T0002\blocknew\ZXG.blk'  # 自选股文件
        except Exception as exception:
            print('未找到tdx配置路径，放弃写入自选股', exception)
            return []

    ret_list = []
    with open(file_name) as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            code = tdxsymbol_to_code(''.join(row))
            if len(code) > 0:
                ret_list.append(code)
    return ret_list


# ================
# MOOTDX
# ================


def get_mootdx_quotes(code_list: list[str]) -> dict[str, any]:
    if code_list is None or len(code_list) == 0:
        return {}

    symbol_list = [code.split('.')[0] for code in code_list]

    mootdx_client = MootdxClientInstance().client
    df = mootdx_client.quotes(symbol=symbol_list)

    result = {}
    for _, row in df.iterrows():
        # 构建股票代码（考虑market字段：0为深交所，1为上交所, 2为北交所）
        market_suffix = '.SZ' if row['market'] == 0 else ('.SH' if row['market'] == 1 else '.BJ')
        stock_code = f"{row['code']}{market_suffix}"

        time_str = row['servertime']    # 转换servertime为毫秒时间戳
        date_str = datetime.datetime.today().strftime('%Y-%m-%d')

        datetime_obj = datetime.datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S.%f')
        timestamp_ms = int(datetime_obj.timestamp() * 1000)

        ask_price = [row[f'ask{i + 1}'] for i in range(5)]
        bid_price = [row[f'bid{i + 1}'] for i in range(5)]
        ask_vol = [row[f'ask_vol{i + 1}'] for i in range(5)]
        bid_vol = [row[f'bid_vol{i + 1}'] for i in range(5)]

        stock_data = {
            'time': timestamp_ms,
            'lastPrice': row['price'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'lastClose': row['last_close'],
            'amount': row['amount'],
            'volume': row['vol'],
            'pvolume': row['vol'] * 100,  # 手转股
            # 'stockStatus': 0,
            # 'openInt': 0,
            # 'transactionNum': 0,
            # 'lastSettlementPrice': 0.0,
            # 'settlementPrice': 0.0,
            # 'pe': 0.0,
            'askPrice': ask_price,
            'bidPrice': bid_price,
            'askVol': ask_vol,
            'bidVol': bid_vol,
            # 'volRatio': 0.0,
            # 'speed1Min': 0.0,
            # 'speed5Min': 0.0
        }
        result[stock_code] = stock_data

    return result


# ================
# TDXZIP
# ================
# 需要安装pycurl，pip install pycurl


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


# ================
# Wencai
# ================


def get_wencai_codes(queries: list[str]) -> list[str]:
    import pywencai
    result = set()
    for query in queries:
        df = pywencai.get(query=query, perpage=100, loop=True)
        if df is not None and type(df) != dict and df.shape[0] > 0:
            result.update(df['股票代码'].values)
    return list(result)


# ================
# Service
# ================


def pull_stock_codes(prefix: str, host: str, auth: str) -> (Optional[list[str]], str):
    key = f'{prefix}_{datetime.datetime.now().date().strftime("%Y%m%d")}'
    response = requests.get(f'{host}/stocks/get_list/{key}?auth={auth}')
    if response.status_code == 200:
        return response.json(), ''
    elif response.status_code == 404:
        return None, response.json()['error']
    else:
        return None, 'Unknown Error'


# ================
# 数据格式处理
# ================


# 数组长度标准化防止quotes数据格式异常导致额外的bug，用以处理买卖五档数据
def _adjust_list(input_list: list, target_length: int) -> list:
    adjusted = input_list[:target_length]               # 截断超过目标长度的尾部
    adjusted += [0] * (target_length - len(adjusted))   # 补0直到达到目标长度
    return adjusted


def qmt_quote_to_tick(quote: dict):
    ans = {
        'time': datetime.datetime.fromtimestamp(quote['time'] / 1000).strftime('%H:%M:%S'),
        'price': quote['lastPrice'],
        'high': quote['high'],
        'low': quote['low'],
        'volume': quote['volume'],
        'amount': quote['amount'],
    }

    ap = _adjust_list(quote['askPrice'], 5)
    ans.update({f"askPrice{i+1}": ap[i] for i in range(min(5, len(ap)))})

    av = _adjust_list(quote['askVol'], 5)
    ans.update({f"askVol{i+1}": av[i] for i in range(min(5, len(av)))})

    bp = _adjust_list(quote['bidPrice'], 5)
    ans.update({f"bidPrice{i+1}": bp[i] for i in range(min(5, len(bp)))})

    bv = _adjust_list(quote['bidVol'], 5)
    ans.update({f"bidVol{i+1}": bv[i] for i in range(min(5, len(bv)))})

    return ans


def qmt_quote_to_day_kline(quote: dict, curr_date: str) -> dict:
    return {
        'datetime': curr_date,
        'open': quote['open'],
        'high': quote['high'],
        'low': quote['low'],
        'close': quote['lastPrice'],
        'volume': quote['volume'],
        'amount': quote['amount'],
    }


# ================
#  AKShare
# ================


def concat_ak_quote_dict(source_df: pd.DataFrame, quote: dict, curr_date: str) -> pd.DataFrame:
    record = qmt_quote_to_day_kline(quote, curr_date=curr_date)
    new_row_df = pd.DataFrame([record.values()], columns=list(record.keys()))
    return pd.concat([source_df, new_row_df], ignore_index=True)


def append_ak_daily_row(source_df: pd.DataFrame, row: dict) -> pd.DataFrame:
    df = source_df._append(row, ignore_index=True)
    return df


def append_ak_spot_dict(source_df: pd.DataFrame, row: pd.Series, curr_date: str) -> pd.DataFrame:
    formatted_row = {
        'datetime': curr_date,
        'open': row['今开'],
        'high': row['最高'],
        'low': row['最低'],
        'close': row['最新价'],
        'volume': row['成交量'],
        'amount': row['成交额'],
    }
    df = append_ak_daily_row(source_df, formatted_row)
    return df


# ================
#  Daily History
# ================


# https://akshare.akfamily.xyz/data/stock/stock.html#id21
def get_ak_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    import akshare as ak
    try:
        if is_stock(code):
            # 东财容易封接口
            # df = ak.stock_zh_a_hist(
            #     symbol=code_to_symbol(code),
            #     start_date=start_date,
            #     end_date=end_date,
            #     adjust=str(adjust),
            #     period='daily',
            # )
            # if len(df) > 0:
            #     df = df.rename(columns={
            #         '日期': 'datetime',
            #         '开盘': 'open',
            #         '最高': 'high',
            #         '最低': 'low',
            #         '收盘': 'close',
            #         '成交量': 'volume',
            #         '成交额': 'amount',
            #     })
            #     df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')
            #     df['datetime'] = df['datetime'].astype(int)

            # 换成新浪的替代
            df = ak.stock_zh_a_daily(
                symbol=code_to_sina_symbol(code),
                start_date=start_date,
                end_date=end_date,
                adjust=str(adjust),
            )
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date']).copy()
            df['datetime'] = df['date'].dt.strftime('%Y%m%d').astype(int)
        elif is_fund_etf(code):
            df = ak.fund_etf_hist_em(
                symbol=code_to_symbol(code),
                start_date=start_date,
                end_date=end_date,
                adjust=str(adjust),
                period="daily",
            )
            if len(df) > 0:
                df = df.rename(columns={
                    '日期': 'datetime',
                    '开盘': 'open',
                    '最高': 'high',
                    '最低': 'low',
                    '收盘': 'close',
                    '成交量': 'volume',
                    '成交额': 'amount',
                })
                df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')
                df['datetime'] = df['datetime'].astype(int)
        else:
            return None
    except Exception as e:
        print(f' akshare get {code} error: ', e)
        return None

    if columns is not None:
        return df[columns]
    return df


def _ts_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        'vol': 'volume',
        'trade_date': 'datetime',
    })
    df['datetime'] = df['datetime'].astype(int)
    df['volume'] = df['volume'].astype(int)
    df['amount'] = df['amount'] * 1000
    df['amount'] = df['amount'].round(2)
    df = df[::-1]
    df.reset_index(drop=True, inplace=True)
    return df


# 使用 tushare 数据源记得 pip install tushare
# 同时配置 tushare 的 token，在官网注册获取
# https://tushare.pro/document/2?doc_id=27
def get_ts_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    # from reader.tushare_agent import get_tushare_pro
    import tushare as ts
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning,
        message=".*Series.fillna with 'method' is deprecated.*")  # 用.*匹配任意字符，关掉tushare内部warning

    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1
        # 奇怪丫又不收费了，复权数据也给了，还不用注册
        # pro = get_tushare_pro()
        # df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        df = ts.pro_bar(ts_code=code, start_date=start_date, end_date=end_date, adj=adjust)
        time.sleep(0.5)

    if df is not None and len(df) > 0:
        df = _ts_to_standard(df)
        if columns is not None:
            return df[columns]
        return df
    return None


# 复合版:通过返回dict的key区分不同的票，注意总共一次最多8000行会限制长度
# https://tushare.pro/document/2?doc_id=27
def get_ts_daily_histories(
    codes: list[str],
    start_date: str,    # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> dict[str, pd.DataFrame]:
    for code in codes:
        if not is_stock(code):
            print(f'存在不符合格式要求的code: {code}')
            return {}

    # from reader.tushare_agent import get_tushare_pro
    import tushare as ts

    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1
        # 同上
        # pro = get_tushare_pro()
        # df = pro.daily(ts_code=','.join(codes), start_date=start_date, end_date=end_date)
        df = ts.pro_bar(ts_code=','.join(codes), start_date=start_date, end_date=end_date, adj=adjust)
        time.sleep(0.5)

    ans = {}
    if df is not None and len(df) > 0:
        for code in codes:
            temp_df = df[df['ts_code'] == code]
            temp_df = _ts_to_standard(temp_df)

            if columns is None:
                ans[code] = temp_df
            else:
                ans[code] = temp_df[columns]
    return ans


def get_bars_with_offset(client, symbol, total_offset, start=0):
    """
    分批获取K线数据，处理offset超过800的限制，并确保时间降序排列

    参数:
        client: 客户端实例
        symbol: 标的代码
        frequency: 频率，默认为'day'
        total_offset: 总共需要获取的K线数量
        start: 起始偏移量

    返回:
        合并并排序后的DataFrame或None
    """
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


# 获取 mootdx 的历史日线
# 使用 mootdx 数据源记得 pip install mootdx
def get_mootdx_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    offset, start = get_offset_start(TRADE_DAY_CACHE_PATH, start_date, end_date)
    symbol = code_to_symbol(code)

    client = MootdxClientInstance().client
    try:
        df = get_bars_with_offset(client, symbol, offset, start)
        # TODO_List: 对于有些期间停牌过的票，发现时间对不上这里要校正，优先级不高因为只会多不会少
    except Exception as e:
        print(f' mootdx get daily {code} error: ', e)
        return None

    if adjust != ExitRight.BFQ:
        xdxr = get_xdxr(symbol=symbol)
        df = execute_fq(df, xdxr, adjust)

    if df is not None and len(df) > 0 and type(df) == pd.DataFrame and 'datetime' in df.columns:
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


def get_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
    data_source=DataSource.AKSHARE,
) -> Optional[pd.DataFrame]:
    if data_source == DataSource.TUSHARE:
        # TuShare 的数据免费的暂时不支持复权
        # TuShare 不支持 etf，其他两个支持
        return get_ts_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.MOOTDX:
        # Mootdx 的复权是先截断数据然后复权，取三位小数
        # 暂时不支持 920xxx 的北交所股票数据
        # 其它北交所股票小部分有发行脏数据情况
        return get_mootdx_daily_history(code, start_date, end_date, columns, adjust)
    else:
        # AkShare 的复权是针对全部历史复权后截取，取两位小数
        # Akshare 的 etf 取三位小数，成交量略有不同
        return get_ak_daily_history(code, start_date, end_date, columns, adjust)


# =======================
#  Convert daily history
# =======================


def convert_daily_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    将日线K线数据转换为周线数据（datetime为该周内最后一个交易日的日期）
    参数: df: 包含日线数据的DataFrame，需包含datetime, open, high, low, close, volume, amount列
    返回: 周线数据的DataFrame，按周一到周日合并，datetime为周内最后一个交易日的日期
    """
    data = df.copy()

    # 1. 保留原始整数日期，同时新增datetime类型列用于分组（确定属于哪一周）
    data['dt'] = pd.to_datetime(data['datetime'], format='%Y%m%d')
    data = data.set_index('dt')  # 用datetime类型索引进行周分组

    # 2. 按周一到周日分组（周区间：[周一, 下周一)）
    weekly_groups = data.resample('W-MON', closed='left', label='left')  # 分组逻辑不变，仅用于划分周范围

    # 3. 聚合规则：核心是对原始datetime取组内最后一个值
    weekly_data = weekly_groups.agg({
        'datetime': 'last',       # 周内最后一个交易日的原始整数日期
        'open': 'first',          # 周内第一个开盘价
        'high': 'max',            # 周内最高价
        'low': 'min',             # 周内最低价
        'close': 'last',          # 周内最后一个收盘价
        'volume': 'sum',          # 周内成交量总和
        'amount': 'sum'           # 周内成交额总和
    }).dropna()  # 移除无数据的周

    # 4. 恢复列顺序和原始数据类型
    weekly_data = weekly_data.reset_index(drop=True)[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    for col in weekly_data.columns:
        weekly_data[col] = weekly_data[col].astype(df[col].dtype)

    return weekly_data


def convert_daily_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    将日线K线数据转换为月线数据（datetime为当月最后一个交易日的日期）
    参数: df: 包含日线数据的DataFrame，需包含datetime, open, high, low, close, volume, amount列
    返回: 月线数据的DataFrame，按自然月划分（1月-12月），datetime为当月最后一个交易日的日期
    """
    data = df.copy()

    # 1. 保留原始整数日期，新增datetime类型列用于按月分组
    data['dt'] = pd.to_datetime(data['datetime'], format='%Y%m%d')
    data = data.set_index('dt')  # 用datetime索引进行月份分组

    # 2. 按自然月分组（1月1日-1月最后一天，2月1日-2月最后一天...）
    # 频率'M'表示按月分组，默认按自然月划分
    monthly_groups = data.resample('M')

    # 3. 聚合规则：与周线逻辑一致，仅周期改为月
    monthly_data = monthly_groups.agg({
        'datetime': 'last',       # 当月最后一个交易日的原始整数日期
        'open': 'first',          # 当月第一个交易日的开盘价
        'high': 'max',            # 当月最高价
        'low': 'min',             # 当月最低价
        'close': 'last',          # 当月最后一个交易日的收盘价
        'volume': 'sum',          # 当月成交量总和
        'amount': 'sum'           # 当月成交额总和
    }).dropna()  # 移除无数据的月份

    # 4. 恢复列顺序和原始数据类型
    monthly_data = monthly_data.reset_index(drop=True)[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    for col in monthly_data.columns:
        monthly_data[col] = monthly_data[col].astype(df[col].dtype)

    return monthly_data
