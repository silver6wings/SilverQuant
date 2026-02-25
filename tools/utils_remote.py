import os
import csv
import time

import requests
import threading
import atexit
from typing import Optional

from tools.constants import DataSource, ExitRight, DEFAULT_DAILY_COLUMNS
from tools.utils_basic import *
from tools.utils_miniqmt import get_qmt_daily_history
from tools.utils_mootdx import MootdxClientInstance, get_mootdx_daily_history


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
# Service
# ================


def pull_stock_codes(prefix: str, host: str, auth: str) -> tuple[Optional[list[str]], str]:
    key = f'{prefix}_{datetime.datetime.now().date().strftime("%Y%m%d")}'
    try:
        response = requests.get(f'{host}/stocks/get_list/{key}?auth={auth}', timeout=10)
    except requests.RequestException as e:
        return None, str(e)

    if response.status_code == 200:
        return response.json(), ''
    if response.status_code == 404:
        response_json = response.json()
        return None, response_json.get('error', 'Not Found')
    return None, f'HTTP {response.status_code}'


# ================
# 数据格式处理
# ================


# 数组长度标准化防止quotes数据格式异常导致额外的bug，用以处理买卖五档数据
def qmt_pad_list(xs, target_length: int, fill=0):
    xs = list(xs) if isinstance(xs, (list, tuple)) else []
    xs = xs[:target_length]
    if len(xs) < target_length:
        xs.extend([fill] * (target_length - len(xs)))
    return xs


def _adjust_list(input_list: list, target_length: int) -> list:
    # 兼容 input_list=None，并保持历史行为：默认补 0
    return qmt_pad_list(input_list, target_length=target_length, fill=0)


def qmt_quote_to_tick(quote: dict) -> dict:
    # 统一做数据兜底：调用方不必再重复补字段
    quote = dict(quote or {})

    # 兼容字段：部分行情里昨收用 lastLose
    if 'lastClose' not in quote:
        quote['lastClose'] = quote.get('lastLose', 0) or 0

    # 兜底基础字段（本函数使用 [] 访问，缺字段会 KeyError）
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

    ans = {
        'time': datetime.datetime.fromtimestamp(quote['time'] / 1000).strftime('%H:%M:%S'),
        'lastClose': quote['lastClose'],
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


ACTIVE_TS_FQ = True  # 是否启用通用行情接口 https://tushare.pro/document/2?doc_id=109


def get_ts_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    from reader.tushare_agent import get_tushare_pro
    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1

        if ACTIVE_TS_FQ:
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning,
                message=".*Series.fillna with 'method' is deprecated.*")  # 用.*匹配任意字符，关掉tushare内部warning

            import tushare as ts
            _ = get_tushare_pro()
            df = ts.pro_bar(ts_code=code, start_date=start_date, end_date=end_date, adj=adjust)
            df = df.drop_duplicates() if df is not None and len(df) > 0 else df
        else:
            pro = get_tushare_pro()
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)

    if df is not None and len(df) > 0:
        df = _ts_to_standard(df)
        if columns is not None:
            return df[columns]
        return df
    return None


# https://tushare.pro/document/2?doc_id=296
# 可惜这个数据要氪金
def get_ts_stk_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    from reader.tushare_agent import get_tushare_pro
    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1

        if adjust == ExitRight.BFQ:
            suffix = ''
        else:
            suffix = f'_{adjust}'

        pro = get_tushare_pro()
        df = pro.stk_factor(
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=','.join([
                'ts_code',
                'trade_date',
                f'open{suffix}',
                f'close{suffix}',
                f'high{suffix}',
                f'low{suffix}',
                f'pre_close{suffix}',
                'vol',
                'amount',
            ])
        )
        df = df.rename(columns={
            f'open{suffix}': 'open',
            f'close{suffix}': 'close',
            f'high{suffix}': 'high',
            f'low{suffix}': 'low',
            f'pre_close{suffix}': 'pre_close',
        })

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
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> dict[str, pd.DataFrame]:
    for code in codes:
        if not is_stock(code):
            print(f'存在不符合格式要求的code: {code}')
            return {}

    from reader.tushare_agent import get_tushare_pro

    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1

        # tushare的通用行情 SDK 有bug，改回去了！
        if ACTIVE_TS_FQ:
            import warnings
            warnings.filterwarnings('ignore', category=FutureWarning,
                message=".*Series.fillna with 'method' is deprecated.*")  # 用.*匹配任意字符，关掉tushare内部warning

            import tushare as ts
            _ = get_tushare_pro()
            df = ts.pro_bar(ts_code=','.join(codes), start_date=start_date, end_date=end_date, adj=adjust)
            df = df.drop_duplicates() if df is not None and len(df) > 0 else df
        else:
            pro = get_tushare_pro()
            df = pro.daily(ts_code=','.join(codes), start_date=start_date, end_date=end_date)

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
        # TuShare 不支持 etf，其他两个支持，但也注意daily_history 不支持 etf
        return get_ts_daily_history(code, start_date, end_date, columns)
    elif data_source == DataSource.MOOTDX:
        # Mootdx 的复权是先截断数据然后复权，取三位小数
        # 暂时不支持 920xxx 的北交所股票数据
        # 其它北交所股票小部分有发行脏数据情况
        return get_mootdx_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.AKSHARE:
        # AkShare 的复权是针对全部历史复权后截取，取两位小数
        # Akshare 的 etf 取三位小数，成交量略有不同
        return get_ak_daily_history(code, start_date, end_date, columns, adjust)
    elif data_source == DataSource.BAOSTOCK:
        return get_bao_daily_history(code, start_date, end_date, columns, adjust)
    else:
        # 默认使用免费的 miniqmt数据，但就是慢的一批
        return get_qmt_daily_history(code, start_date, end_date, columns, adjust)
