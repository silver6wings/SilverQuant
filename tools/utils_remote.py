import csv
import datetime
import requests
import numpy as np
import pandas as pd
from typing import Optional

from tools.utils_basic import is_stock, code_to_symbol, tdxsymbol_to_code, code_to_tdxsymbol
from tools.utils_cache import TRADE_DAY_CACHE_PATH
from tools.utils_mootdx import MootdxClientInstance, get_offset_start, make_qfq, make_hfq

from credentials import TDX_FOLDER


DEFAULT_ZXG_FILE = TDX_FOLDER + r'\T0002\blocknew\ZXG.blk'     # 自选股文件


class DataSource:
    AKSHARE = 'akshare'
    TUSHARE = 'tushare'
    MOOTDX = 'mootdx'


class ExitRight:
    BFQ = ''     # 不复权
    QFQ = 'qfq'  # 前复权
    HFQ = 'hfq'  # 后复权


def set_tdx_zxg_code(data: list[str], filename: str = DEFAULT_ZXG_FILE):
    # 打开或创建CSV文件并指定写入模式, newline=''则不生成空行
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([code_to_tdxsymbol(item)])
    print(f"已成功将数据写入{filename}文件！")


def get_tdx_zxg_code(filename: str = DEFAULT_ZXG_FILE) -> list[str]:
    ret_list = []
    with open(filename) as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            ret_list.append(tdxsymbol_to_code(''.join(row)))
    return ret_list


# ================
# MOOTDX
# ================


def get_mootdx_quotes(code_list: list[str]):
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
        'volume': quote['volume'],
        'amount': quote['amount'],
        'high': quote['high'],
        'low': quote['low'],
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
    if not is_stock(code):
        return None

    import akshare as ak
    try:
        df = ak.stock_zh_a_hist(
            symbol=code_to_symbol(code),
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
            period='daily',
        )
    except Exception as e:
        print(f' akshare get {code} error: ', e)
        df = []

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
        if columns is not None:
            return df[columns]
        return df
    return None


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
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    from reader.tushare_agent import get_tushare_pro
    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        pro = get_tushare_pro()
        try_times += 1
        df = pro.daily(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
        )

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
) -> dict[str, pd.DataFrame]:
    from reader.tushare_agent import get_tushare_pro
    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        pro = get_tushare_pro()
        try_times += 1
        df = pro.daily(
            ts_code=','.join(codes),
            start_date=start_date,
            end_date=end_date,
        )

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


# 获取 mootdx 的历史日线
# 使用 mootdx 数据源记得 pip install mootdx
def get_mootdx_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    offset, start = get_offset_start(TRADE_DAY_CACHE_PATH, start_date, end_date)
    symbol = code_to_symbol(code)

    client = MootdxClientInstance().client
    try:
        df = client.bars(
            symbol=symbol,
            frequency='day',
            offset=offset,  # 总共N个K线
            start=start,    # 向前数跳过几行
        )
        df = df.replace([np.inf, -np.inf], np.nan)
        # 然后删除所有包含NaN的行

        # print(df.dtypes)
        # print(df)
    except Exception as e:
        print(f' mootdx get daily {code} error: ', e)
        return None

    if adjust != ExitRight.BFQ:
        try:
            xdxr = client.xdxr(symbol=symbol)
            if xdxr is not None and len(xdxr) > 0:
                xdxr['date_str'] = xdxr['year'].astype(str) + \
                    '-' + xdxr['month'].astype(str).str.zfill(2) + \
                    '-' + xdxr['day'].astype(str).str.zfill(2)
                xdxr['datetime'] = pd.to_datetime(xdxr['date_str'] + ' 15:00:00')
                xdxr = xdxr.set_index('datetime')

                if adjust == ExitRight.QFQ:
                    df = make_qfq(df, xdxr)
                elif adjust == ExitRight.HFQ:
                    df = make_hfq(df, xdxr)
        except Exception as e:
            print(f' mootdx get xdxr {code} error: ', e)
            return None

    if df is not None and len(df) > 0 and type(df) == pd.DataFrame and 'datetime' in df.columns:
        try:
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['datetime'] = df['datetime'].dt.date.astype(str).str.replace('-', '').astype(int)
            df = pd.concat([df['datetime'], df.drop('datetime', axis=1)], axis=1)

            df = df.drop(columns='volume')
            df = df.rename(columns={'vol': 'volume'})
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
        # Tushare 的数据免费的暂时不支持复权
        return get_ts_daily_history(code, start_date, end_date, columns)
    elif data_source == DataSource.MOOTDX:
        # Mootdx 的复权是先截断数据然后复权，取三位小数
        # 暂时不支持 920xxx 的北交所股票数据
        # 其它北交所股票小部分有发行脏数据情况
        return get_mootdx_daily_history(code, start_date, end_date, columns, adjust)
    else:
        # Akshare 的复权是针对全部历史复权后截取，取两位小数
        return get_ak_daily_history(code, start_date, end_date, columns, adjust)
