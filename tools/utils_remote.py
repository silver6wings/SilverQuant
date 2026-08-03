import os
import csv
import datetime
import time

import pandas as pd
import requests
import threading
import atexit
from typing import Optional

from tools.constants import DataSource, ExitRight, DEFAULT_DAILY_COLUMNS
from tools.utils_basic import code_to_symbol, code_to_sina_symbol, code_to_tdxsymbol, \
    is_fund_etf, is_stock, tdxsymbol_to_code

from tools.utils_remote_ts import get_ts_daily_histories, get_ts_daily_history, get_ts_stk_daily_history
from tools.utils_remote_sv import pull_stock_today_codes, push_stock_today_codes
from delegate.daily_reporter import colour_text


class BaoStockInstance:
    _instance = None
    _lock = threading.Lock()
    bs = None
    _initialized = False
    _login_ok = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BaoStockInstance, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            import baostock as bs
            self.bs = bs
            print('[BAOSTOCK] login...', end='')
            lg = bs.login()
            if lg.error_code == '0':
                self._login_ok = True
            else:
                print('[BAOSTOCK] login respond error_msg: ' + lg.error_msg)
            atexit.register(self.logout)
            self._initialized = True

    def logout(self):
        if self.bs is not None and self._login_ok:
            try:
                print('[BAOSTOCK] logout...', end='')
                self.bs.logout()
            except Exception as e:
                print('[BAOSTOCK] logout error!', e)
            self._login_ok = False

    def __del__(self):
        self.logout()


def set_tdx_zxg_code(data: list[str], file_name: str = None, block_name: str = '自选股') -> None:
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
    print(f'已成功将数据写入{block_name}文件：{file_name}')


def get_tdx_zxg_code(file_name: str = None) -> list[str]:
    if file_name is None:
        try:
            from credentials import TDX_FOLDER
            file_name = TDX_FOLDER + r'\T0002\blocknew\ZXG.blk'  # 自选股文件
        except Exception as exception:
            print('未找到tdx配置路径，放弃写入自选股', exception)
            return []
   
    ret_list = []
    if os.path.isfile(file_name):
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

    from tools.utils_mootdx import MootdxClientInstance

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
        df = None
        try:
            df = pywencai.get(query=query, perpage=100, loop=True)
        except Exception as e:
            print('获取wencai数据失败,请尝试降低获取频率:', e)

        if df is not None and type(df) != dict and df.shape[0] > 0:
            result.update(df['股票代码'].values)

    return list(result)


# ================
# QMT Quote 数据格式处理
# ================


def is_tick_quote(data) -> bool:
    """sickle 统一 tick quote（timestamp + 扁平五档，见 sickle/data/tick/tick_quote.py）。"""
    return isinstance(data, dict) and 'timestamp' in data


def tick_quote_to_qmt_quote(quote: dict) -> dict:
    """统一 tick quote -> QMT subscribe_whole_quote 字段（time + 五档 list）。"""
    q = quote or {}
    return {
        'time': int(q.get('timestamp', 0) or 0),
        'lastClose': q.get('lastClose', 0) or 0,
        'open': q.get('open', 0) or 0,
        'high': q.get('high', 0) or 0,
        'low': q.get('low', 0) or 0,
        'lastPrice': q.get('lastPrice', 0) or 0,
        'volume': q.get('volume', 0) or 0,
        'amount': q.get('amount', 0) or 0,
        'askPrice': [float(q.get(f'askPrice{i}', 0) or 0) for i in range(1, 6)],
        'bidPrice': [float(q.get(f'bidPrice{i}', 0) or 0) for i in range(1, 6)],
        'askVol': [int(q.get(f'askVol{i}', 0) or 0) for i in range(1, 6)],
        'bidVol': [int(q.get(f'bidVol{i}', 0) or 0) for i in range(1, 6)],
    }


# tick 统一列定义（list 行 / DataFrame / parquet 共用，改列只改此处）
QMT_TICK_DF_COLS: tuple[str, ...] = (
    ['local', 'time', 'price', 'high', 'low', 'lastClose', 'volume', 'amount']
    + [f'askPrice{i}' for i in range(1, 6)]
    + [f'askVol{i}' for i in range(1, 6)]
    + [f'bidPrice{i}' for i in range(1, 6)]
    + [f'bidVol{i}' for i in range(1, 6)]
)

TICK_LIST_LEN = len(QMT_TICK_DF_COLS)
TICK_LIST_COL_LOCAL = QMT_TICK_DF_COLS.index('local')
TICK_LIST_COL_TIME = QMT_TICK_DF_COLS.index('time')
TICK_LIST_COL_PRICE = QMT_TICK_DF_COLS.index('price')


def tick_list_row_time(row) -> str:
    return row[TICK_LIST_COL_TIME]


def tick_list_row_price(row) -> float:
    return row[TICK_LIST_COL_PRICE]


def _normalize_qmt_quote(quote: dict) -> dict:
    quote = dict(quote or {})
    if 'lastClose' not in quote:
        quote['lastClose'] = quote.get('lastLose', 0) or 0
    quote.setdefault('time', int(time.time() * 1000))
    quote.setdefault('lastPrice', 0)
    quote.setdefault('high', 0)
    quote.setdefault('low', 0)
    quote.setdefault('volume', 0)
    quote.setdefault('amount', 0)
    quote.setdefault('askPrice', [])
    quote.setdefault('askVol', [])
    quote.setdefault('bidPrice', [])
    quote.setdefault('bidVol', [])
    return quote


def _quote_tick_time_hms(quote: dict, fallback: str) -> str:
    tick_ts_ms = quote.get('time')
    if isinstance(tick_ts_ms, (int, float)) and tick_ts_ms > 0:
        return datetime.datetime.fromtimestamp(tick_ts_ms / 1000).strftime('%H:%M:%S')
    return fallback


# 数组长度标准化防止quotes数据格式异常导致额外的bug，用以处理买卖五档数据
def qmt_pad_list(xs, target_length: int, fill=0):
    xs = list(xs) if isinstance(xs, (list, tuple)) else []
    xs = xs[:target_length]
    if len(xs) < target_length:
        xs.extend([fill] * (target_length - len(xs)))
    return xs


def _qmt_quote_to_tick_record(quote: dict, local_time: Optional[str] = None) -> dict:
    """QMT quote -> 标准 tick 字典（键与 QMT_TICK_DF_COLS 一致）。"""
    local_time = local_time or datetime.datetime.now().strftime('%H:%M:%S')
    quote = _normalize_qmt_quote(quote)
    tick_time = _quote_tick_time_hms(quote, local_time)

    ask_price = qmt_pad_list(quote.get('askPrice', []), target_length=5, fill=0.0)
    ask_vol = qmt_pad_list(quote.get('askVol', []), target_length=5, fill=0)
    bid_price = qmt_pad_list(quote.get('bidPrice', []), target_length=5, fill=0.0)
    bid_vol = qmt_pad_list(quote.get('bidVol', []), target_length=5, fill=0)

    rec = {
        'local': local_time,
        'time': tick_time,
        'price': round(float(quote.get('lastPrice', 0) or 0), 3),
        'high': round(float(quote.get('high', 0) or 0), 3),
        'low': round(float(quote.get('low', 0) or 0), 3),
        'lastClose': round(float(quote.get('lastClose', 0) or 0), 3),
        'volume': int(quote.get('volume', 0) or 0),
        'amount': round(float(quote.get('amount', 0) or 0), 3),
    }
    for i in range(5):
        rec[f'askPrice{i + 1}'] = round(float(ask_price[i] or 0.0), 3)
        rec[f'askVol{i + 1}'] = int(ask_vol[i] or 0)
        rec[f'bidPrice{i + 1}'] = round(float(bid_price[i] or 0.0), 3)
        rec[f'bidVol{i + 1}'] = int(bid_vol[i] or 0)
    return rec


def qmt_quote_to_tick_list_row(quote: dict, local_time: Optional[str] = None) -> list:
    """QMT quote -> 拍平 tick 行（列序与 QMT_TICK_DF_COLS 一致）。"""
    rec = _qmt_quote_to_tick_record(quote, local_time=local_time)
    return [rec[col] for col in QMT_TICK_DF_COLS]


def tick_list_row_to_record(row) -> dict:
    """拍平 tick 行 -> 标准 tick 字典。"""
    return dict(zip(QMT_TICK_DF_COLS, row))


def qmt_quote_to_tick(quote: dict) -> dict:
    """QMT quote -> tick 字典（不含 local，供 am_subscriber 等写入时自行追加）。"""
    rec = _qmt_quote_to_tick_record(quote)
    return {col: rec[col] for col in QMT_TICK_DF_COLS if col != 'local'}


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
    return pd.concat([source_df, new_row_df], ignore_index=True) if len(source_df) > 0 else new_row_df


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
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
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


# http://www.baostock.com
def get_bao_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    start = f"{str(start_date)[:4]}-{str(start_date)[4:6]}-{str(start_date)[6:]}"
    end = f"{str(end_date)[:4]}-{str(end_date)[4:6]}-{str(end_date)[6:]}"

    bs = BaoStockInstance().bs

    adjust_flag = '3'
    if adjust == ExitRight.QFQ:
        adjust_flag = '2'
    elif adjust == ExitRight.HFQ:
        adjust_flag = '1'

    [symbol, exchange] = code.split('.')
    rs = bs.query_history_k_data_plus(
        f'{exchange.lower()}.{symbol}',
        "date,code,open,high,low,close,volume,amount,peTTM",
        start_date=start,
        end_date=end,
        frequency='d',
        adjustflag=adjust_flag,
    )
    if rs.error_code == '0':
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
        df = df.rename(columns={'date': 'datetime'})
        df['datetime'] = df['datetime'].astype(int)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].replace('', '0').astype(int) / 100  # 停牌的票会返回空串所以改成0
        df['amount'] = df['amount'].replace('', '0').astype(float)

        if df is not None and len(df) > 0:
            if columns is not None:
                return df[columns]
            return df
        return None
    else:
        print(f'query_history_k_data_plus {code} respond error_msg:' + rs.error_msg)
        return None


def get_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
    data_source: DataSource = DataSource.TUSHARE,
) -> Optional[pd.DataFrame]:
    if data_source == DataSource.TUSHARE:
        # TuShare 不支持 etf，其他两个支持，但也注意 daily_history 不支持 etf
        return get_ts_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.MOOTDX:
        # Mootdx 的复权是先截断数据然后复权，取三位小数
        # 暂时不支持 920xxx 的北交所股票数据
        # 其它北交所股票小部分有发行脏数据情况
        from tools.utils_mootdx import get_mootdx_daily_history
        return get_mootdx_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.AKSHARE:
        # AkShare 的复权是针对全部历史复权后截取，取两位小数
        # Akshare 的 etf 取三位小数，成交量略有不同
        return get_ak_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.BAOSTOCK:
        return get_bao_daily_history(code, start_date, end_date, columns, adjust)
    else:
        # 默认使用免费的 miniqmt数据，但就是慢的一批
        from tools.utils_remote_xt import get_qmt_daily_history
        return get_qmt_daily_history(code, start_date, end_date, columns, adjust)


# 同花顺概念板块排名
THS_CONCEPT_KEYS = [
    ['即时', '净额', '当日', '行业-涨跌幅'],
    ['3日排行', '净额', '三日', '阶段涨跌幅'],
    ['5日排行', '净额', '五日', '阶段涨跌幅'],
    ['10日排行', '净额', '十日', '阶段涨跌幅'],
    ['20日排行', '净额', '二十日', '阶段涨跌幅'],
]


def get_ths_concept_ranking_df(
    *,
    key_index: int = 0,
    is_outflow: bool = False,
) -> pd.DataFrame:
    import akshare as ak

    key = THS_CONCEPT_KEYS[key_index]

    df = ak.stock_fund_flow_concept(symbol=key[0])
    df = df[df['公司家数'] <= 600]
    df = df.sort_values(by='净额', ascending=is_outflow)

    if key[0] != '即时':
        df[key[1]] = df[key[3]].str.strip('%').astype(float)

    return df.sort_values(by=key[1], ascending=is_outflow)


def get_ths_concept_ranking_str(
    *,
    up_df: pd.DataFrame = None,
    key_index: int = 0,
    top_n: int = 10,
    is_outflow: bool = False,
) -> str:
    if up_df is None:
        up_df = get_ths_concept_ranking_df(key_index=key_index, is_outflow=is_outflow)
    
    key = THS_CONCEPT_KEYS[key_index]

    direction_text = '流出' if is_outflow else '流入'
    ans = f'同花顺{key[2]}净{direction_text}前{top_n}概念板块\n'

    name = up_df.head(top_n)['行业'].values
    rate = up_df.head(top_n)[[key[1]]].values
    if len(name) == 0:
        return ans

    longest_name = len(max(name, key=len))
    for i in range(len(name)):
        amount = float(rate[i][0])
        amount_text = colour_text(f'{amount}亿', to_red=amount > 0, to_green=amount < 0)
        ans += f'[{i + 1}]\t{name[i]} ' \
               f'{" " * ((longest_name - len(name[i])) * 1)}' \
               f'\t{amount_text}\n'

    return ans


def get_ths_industry_ranking_df(
    *,
    key_index: int = 0,
    is_fall: bool = False,
) -> pd.DataFrame:
    import akshare as ak

    key = THS_CONCEPT_KEYS[key_index]
    rate_col = key[3]

    df = ak.stock_fund_flow_industry(symbol=key[0])
    if key[0] != '即时':
        df[rate_col] = df[rate_col].str.strip('%').astype(float)
    else:
        df[rate_col] = df[rate_col].astype(float)

    return df.sort_values(by=rate_col, ascending=is_fall)


def get_ths_industry_ranking_str(
    *,
    up_df: pd.DataFrame = None,
    key_index: int = 0,
    top_n: int = 5,
    is_fall: bool = False,
) -> str:
    if up_df is None:
        up_df = get_ths_industry_ranking_df(key_index=key_index, is_fall=is_fall)

    key = THS_CONCEPT_KEYS[key_index]
    rate_col = key[3]

    rise_or_fall = '跌幅' if is_fall else '涨幅'
    ans = f'同花顺{key[2]}{rise_or_fall}前{top_n}行业板块：\n'

    name = up_df.head(top_n)['行业'].values
    rate = up_df.head(top_n)[rate_col].values
    if len(name) == 0:
        return ans

    longest_name = len(max(name, key=len))
    for i in range(len(name)):
        pct = float(rate[i])
        pct_text = colour_text(f'{pct:.2f} %', to_red=pct > 0, to_green=pct < 0)
        ans += f'[{i + 1}]\t{name[i]} {" " * ((longest_name - len(name[i])) * 2)}\t{pct_text}\n'

    return ans
