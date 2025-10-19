import os
import sys
import csv
import time
import datetime
import io
import json
import logging
import traceback
import zipfile
import pycurl

import requests
import numpy as np
import pandas as pd
from typing import Optional

from tools.utils_basic import is_stock, is_fund_etf, code_to_symbol, tdxsymbol_to_code, code_to_tdxsymbol
from tools.utils_cache import TRADE_DAY_CACHE_PATH, get_available_stock_codes, get_prev_trading_date_list, get_trading_date_list, load_pickle, save_pickle
from tools.utils_mootdx import MootdxClientInstance, get_offset_start, make_qfq, make_hfq, get_xdxr
from tools.utils_mootdx import MootdxDailyBarReaderInstance, _fetch_xdxr_factor, download_tdx_hsjday, factor_reversion, get_xdxr_sina


PATH_TDX_HISTORY = f'./_cache/_daily_tdxzip/history_tdxhsj.pkl'
PATH_TDX_XDXR = f'./_cache/_daily_tdxzip/xdxr.pkl'

class DataSource:
    AKSHARE = 'akshare'
    TUSHARE = 'tushare'
    MOOTDX = 'mootdx'
    TDXZIP = 'tdxzip'


class ExitRight:
    BFQ = ''     # 不复权
    QFQ = 'qfq'  # 前复权
    HFQ = 'hfq'  # 后复权

try:
    from credentials import TDX_FOLDER
    DEFAULT_TDX_PATH = TDX_FOLDER + r'\T0002\blocknew\ZXG.blk'  # 自选股文件
except:
    DEFAULT_TDX_PATH = None


def set_tdx_zxg_code(data: list[str], filename: str = DEFAULT_TDX_PATH) -> None:
    if filename is None:
        print(f'路径为空，放弃写入自选股')
        return

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for item in data:
            writer.writerow([code_to_tdxsymbol(item)])
    print(f'已成功将数据写入自选股文件：{filename}')


def get_tdx_zxg_code(filename: str = DEFAULT_TDX_PATH) -> list[str]:
    if filename is None:
        return []

    ret_list = []
    with open(filename) as f:
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

def get_tdxzip_history(adjust=ExitRight.QFQ, day_count=550 ) -> dict|None:
    '''
    直接从通达信网站下载日线文件加载到cache_history，且完成前复权计算
    TDX hsjday.zip 缓存在./_cache/_daily_tdxzip/ 目录下，除权除息缓存文件也在xdxr.pkl目录下
    '''
    cache_history = {}
    cachehistory_path = PATH_TDX_HISTORY
    cachepath = os.path.dirname(cachehistory_path)
    os.makedirs(cachepath, exist_ok=True)
    if os.path.isfile(cachehistory_path) and os.path.getmtime(cachehistory_path) > time.mktime(datetime.date.today().timetuple()): #当天文件才加载
        cache_history = load_pickle(cachehistory_path)
        return cache_history
    
    code_list = get_available_stock_codes()

    print(f'[HISTORY] Downloading {len(code_list)} gap codes data of {day_count} days.')
    tdx_hsjday_file = f'{cachepath}/hsjday.zip'
    cache_xdxr_file = PATH_TDX_XDXR
    try:
        if not os.path.exists(tdx_hsjday_file) or os.path.getmtime(tdx_hsjday_file) < time.mktime(datetime.date.today().timetuple()):
            start = time.time()
            buffer = download_tdx_hsjday()
            end = time.time()
            if buffer == False:
                return cache_history
            buffer.seek(0, io.SEEK_END)
            response_length = buffer.tell()
            filesize = response_length / 1024 / 1024
            print(f'[HISTORY] 下载通达信日线文件耗时{end-start:.2f}秒, 速度：{filesize/(end-start):.2f} MB/s。')
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
                        if str(error_type).startswith('xdxr erro'):
                            download_failure.append(code)
                            print(f'{code}: {error_type}')
                    else:
                        download_failure.append(code)
                        print(f'{code}: {error_type}')
                except Exception as e:
                    download_failure.append(code)
        
        error_count = len(download_failure)
        
        end = time.time()
        buffer.close()
        print(f'[HISTORY] Download finished with {downloaded_count} code, Elapsed time: {end-start:.2f}s, {error_count} errors and failed with {len(download_failure)} fails: {download_failure}')
        if len(cache_xdxr) > 0:
            save_pickle(PATH_TDX_XDXR, cache_xdxr)
        del buffer
        save_pickle(PATH_TDX_HISTORY, cache_history)
        return True
    except Exception as e:
        print(f'[HISTORY] get tdx hsjday date error :  ',e)
        return False

def _process_tdx_zip_to_datas(groupcodes, zip_ref, cache_xdxr, day_count, adjust):
    """处理tdx zip文件，未来需改造多线程"""
    now = datetime.datetime.now()
    curr_date = now.strftime("%Y-%m-%d")
    result = {}
    factor_name = f'{adjust}_factor'
    default_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount', 'adj']
    barreader = MootdxDailyBarReaderInstance().reader
    
    for code in groupcodes:
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
                #以下代码来自 TdxDailyBarReader.get_df_by_file()
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
                xdxr = get_xdxr_sina(symbol, adjust, factor_name)
                if xdxr is not None and len(xdxr) > 0:
                    cache_xdxr[code] = xdxr

            if xdxr is not None and len(xdxr) > 0 and adjust and adjust in [ExitRight.QFQ, ExitRight.HFQ] :
                
                try:
                    if symbol == '689009':
                        raise Exception('CDR公司') 
                    if factor_name not in xdxr.columns: # 没有复权因子数据需要更新
                        fq = _fetch_xdxr_factor(symbol, adjust, factor_name)
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
                    df = factor_reversion(adjust, df, fq)
                    df.rename(columns={'factor': 'adj'}, inplace=True)
                except Exception as e:
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
                    if is_append == True:
                        df = df[:-1]
        except Exception as e:
            result[code] = df, code, f'xdxr error: str{e}'
            continue

        if df is not None and len(df) > 0 and type(df) == pd.DataFrame and 'datetime' in df.columns:
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
                print('x', end='')
                result[code] = None, code, 'format error: str{e}'
                continue
        else:
            result[code] = None, code, 'no data'
            continue
    return result

# 检查xdxr缓存，建议定时调度运行
def check_xdxr_cache(adjust=ExitRight.QFQ) -> None:
    '''
    检查缓存的通达信除权除息数据，cache_xdxr数据格式为stockcode:pd.DataFrame，以及updatedtime: Datetime
    如果发现有除权除息日大于updatetime 且小于当前日期，就删除cache条目
    '''
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
            date_list = get_trading_date_list(updated_date, curr_date)[1:]

        factor_name = f'{adjust}_factor'
        removed_xdxr_codes = set()

        for cdate in date_list:
            try:
                xcdf, divicount = _get_dividend_code_from_baidu(cdate)  #该接口返回的df中code实际上是symbol，注意转换
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

        if len(removed_xdxr_codes)>0:
            print(f'{len(removed_xdxr_codes)}只股票需要更新复权因子。')
            succ_count = 0
            for symbol in removed_xdxr_codes:
                code = symbol_to_code(symbol)
                try:
                    xdxr = get_xdxr_sina(symbol, adjust, factor_name)
                    if xdxr is not None and not xdxr.empty:
                        curr_xc = xdxr.loc[xdxr['category'] == 1]
                        if not curr_xc.empty and factor_name in curr_xc.columns and float(curr_xc.iloc[-1][factor_name]) == 1.0 :
                            cache_xdxr[code] = xdxr
                            succ_count += 1
                            continue
                    logging.error(f'更新{code}除权除息数据有异常，无有效数据。')
                    print(f'更新{code}除权除息数据有异常，无有效数据。')
                except Exception as e:
                    logging.error(f'更新{code}除权除息数据失败，错误:{str(e)}')
                    print(f'更新{code}除权除息数据失败，错误:{str(e)}')
            if len(removed_xdxr_codes) == succ_count: # 成功更新才更新时间，让第二天再次更新
                cache_xdxr['updatedtime'] = datetime.datetime.now()
            save_pickle(PATH_TDX_XDXR, cache_xdxr)
            print(f'成功更新{succ_count}只股票复权因子。')
    except Exception as e: #异常不要紧，不要因为异常影响实际运行
        logging.error(f'处理除权除息数据出现问题：{str(e)}')

def _pycurl_request(url, headers=None):
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

def _get_dividend_code_from_baidu(start_date: str = "20241107") -> (pd.DataFrame, int):
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

    def _get_stock_dividend(start_date: str, page=0) -> pd.DataFrame:
        divi_df, divi_count, has_more = pd.DataFrame(), 0, False
        if len(start_date) == 8:  # 20241107 -> 2024-11-07
            start_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
        try:
            url = f"https://finance.pae.baidu.com/sapi/v1/financecalendar?start_date={start_date}&end_date={start_date}&market=ab&pn={page}&rn=100&cate=notify_divide&finClientType=pc"
            headers.update({'Accept': 'application/vnd.finance-web.v1+json'})
            status_code, content = _pycurl_request(url, headers=headers)
            if status_code != 200:
                raise Exception(f'百度股市通接口异常，返回HTTP 状态码为{status_code}')
                return divi_df, divi_count, has_more
            data_json = json.loads(content)
            if data_json.get('status', 0)!=0 or data_json.get('ResultCode', -1) != 0: #出错了
                raise Exception(f'百度股市通接口异常，返回HTTP 状态码为{data_json.get('status', 0)} {data_json.get('ResultCode', -1)}')
                return divi_df, divi_count, has_more

            for item in data_json['Result']['calendarInfo']:
                divi_data = item["list"]
                divi_count += item['total']
                has_more = bool(item['hasMore'])
                if len(divi_data) > 0:
                    divi_df = pd.concat([divi_df, pd.DataFrame(divi_data)], ignore_index=True)

            return divi_df[["code","name","date"]], divi_count, has_more #
        except Exception as e:
            raise Exception(f'百度股市通接口异常：{str(e)}')
            return divi_df, divi_count, has_more

    #  函数主体
    headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36 Edg/141.0.0.0',
                'Referer': 'https://gushitong.baidu.com/',
                'Origin': 'https://gushitong.baidu.com',
            }
    url = 'https://gushitong.baidu.com/calendar'
    status_code, content = _pycurl_request(url, headers=headers)
    page = 0
    resdf = []
    while True:
        df, divi_count, has_more = _get_stock_dividend(start_date, page)
        if len(resdf) == 0 and has_more == False: #第一次调用的时候如果没有更多分页直接返回
            return df, divi_count
        if not df.empty:
            resdf.append(df)

        if has_more == False:
            return pd.concat(resdf, ignore_index=True), divi_count  
        page +=1
        time.sleep(0.5)


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
            df = ak.stock_zh_a_hist(
                symbol=code_to_symbol(code),
                start_date=start_date,
                end_date=end_date,
                adjust=str(adjust),
                period='daily',
            )
        elif is_fund_etf(code):
            df = ak.fund_etf_hist_em(
                symbol=code_to_symbol(code),
                start_date=start_date,
                end_date=end_date,
                adjust=str(adjust),
                period="daily",
            )
        else:
            return None
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
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    if not is_stock(code):
        return None

    # from reader.tushare_agent import get_tushare_pro
    import tushare as ts
    pd.set_option('future.no_silent_downcasting', True)

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
            return None

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


def execute_fq(df, xdxr, adjust):
    if xdxr is not None and len(xdxr) > 0:
        xdxr['date_str'] = xdxr['year'].astype(str) + \
                           '-' + xdxr['month'].astype(str).str.zfill(2) + \
                           '-' + xdxr['day'].astype(str).str.zfill(2)
        xdxr['datetime'] = pd.to_datetime(xdxr['date_str'] + ' 15:00:00')
        xdxr = xdxr.set_index('datetime')

        is_appended = False
        xdxr_info = xdxr.loc[xdxr['category'] == 1]

        now = datetime.datetime.now()
        curr_date = now.strftime("%Y-%m-%d")

        # 默认除权日当天之前的数据一样进行处理
        if not xdxr_info.empty and xdxr_info.index[-1].date() <= now.date():
            last_row = df.iloc[-1].copy()
            last_row['datetime'] = curr_date
            df.loc[len(df)] = last_row
            df.index = pd.to_datetime(df['datetime'].astype(str), errors="coerce")
            is_appended  = True

        if adjust == ExitRight.QFQ:
            df = make_qfq(df, xdxr)
        elif adjust == ExitRight.HFQ:
            df = make_hfq(df, xdxr)

        if is_appended:
            df = df[:-1]

    return df


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
