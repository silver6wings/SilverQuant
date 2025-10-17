import io
import logging
import os
import sys
import time
import datetime
import zipfile
import numpy as np
import pandas as pd
import pycurl

from mootdx.utils import factor
from mootdx.consts import MARKET_BJ, MARKET_SH, MARKET_SZ
from tdxpy.constants import SECURITY_EXCHANGE
from tdxpy.reader import TdxDailyBarReader

DEFAULT_XDXR_CACHE_PATH = './_cache/_daily_mootdx/xdxr'


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
            try:
                from credentials import TDX_FOLDER
                self.client = Quotes.factory(market='std', tdxdir=TDX_FOLDER)
            except Exception as e:
                print('未找到本地TDX目录，使用默认TDX数据源配置：', e)
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



def get_xdxr(symbol: str, cache_dir: str = DEFAULT_XDXR_CACHE_PATH, expire_hours: int = 12):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{symbol}.csv")  # 缓存文件名：股票代码.csv
    expire_seconds = expire_hours * 3600

    if os.path.exists(cache_file):
        file_mtime = os.path.getmtime(cache_file)
        time_diff = time.time() - file_mtime

        if time_diff <= expire_seconds:
            return pd.read_csv(cache_file)

    client = MootdxClientInstance().client
    xdxr_data = client.xdxr(symbol=symbol)

    if xdxr_data is not None:  # 简单判断数据有效性
        xdxr_data.to_csv(cache_file, index=False)  # index=False不保存索引列

    return xdxr_data


def get_offset_start(csv_path: str, start_date_str: str, end_date_str: str) -> tuple[int, int]:
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

    # 清理数据, 返回结果
    return data.query("open != 0").drop(
        [
            "fenhong",
            "peigu",
            "peigujia",
            "songzhuangu",
        ],
        axis=1,
    )


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

    return data.query('if_trade==1 and open != 0').drop(
        ['fenhong', 'peigu', 'peigujia', 'songzhuangu', 'if_trade', 'category'], axis=1)

def _get_stock_market(symbol='', string=False):
    """ 本函数从mootdx而来，增加北交所判断，mootdx修复后需改为另外调用
    判断股票ID对应的证券市场匹配规则

    ['50', '51', '60', '90', '110'] 为 sh
    ['00', '12'，'13', '18', '15', '16', '18', '20', '30', '39', '115'] 为 sz
    ['5', '6', '9'] 开头的为 sh， 其余为 sz

    :param string: False 返回市场ID，否则市场缩写名称
    :param symbol: 股票ID, 若以 'sz', 'sh' 开头直接返回对应类型，否则使用内置规则判断
    :return 'sh' or 'sz'
    """

    assert isinstance(symbol, str), 'stock code need str type'

    market = 'sh'

    if symbol.startswith(('sh', 'sz', 'SH', 'SZ', 'bj', 'BJ')):
        market = symbol[:2].lower()

    elif symbol.startswith(('50', '51', '58', '60', '68', '90', '110', '111', '113', '118', '240')):
        market = 'sh'

    elif symbol.startswith(('00', '12', '13', '18', '15', '16', '18', '20', '30', '39', '115')):
        market = 'sz'

    elif symbol.startswith(('5', '6', '7', '90', '88', '98', '99')):
        market = 'sh'

    elif symbol.startswith(('20', '4', '82', '83', '87', '92')):
        market = 'bj'


    # logger.debug(f"market => {market}")

    if string is False:
        if market == 'sh':
            market = MARKET_SH

        if market == 'sz':
            market = MARKET_SZ

        if market == 'bj':
            market = MARKET_BJ

    # logger.debug(f"market => {market}")

    return market

def _fetch_xdxr_factor(symbol, adjust, factor_name=None) -> pd.DataFrame:
    if factor_name is None:
        factor_name = f'{adjust}_factor'
    market = _get_stock_market(symbol, string=True)
    cache_file = factor.get_config_path(f'caches/factor/{market}{symbol}_{adjust}.plk')
    if os.path.isfile(cache_file):
        os.remove(cache_file)
    fq = factor.fq_factor(symbol, adjust)
    fq.rename(columns={'factor': factor_name}, inplace=True)
    fq.sort_index(ascending=True, inplace=True)
    return fq
        
def get_xdxr_sina(symbol, adjust, factor_name=None) -> pd.DataFrame:
    xdxr = MootdxClientInstance().client.xdxr(symbol=symbol)
    if xdxr is not None and len(xdxr) > 0:
        xdxr['date_str'] = xdxr['year'].astype(str) + \
            '-' + xdxr['month'].astype(str).str.zfill(2) + \
            '-' + xdxr['day'].astype(str).str.zfill(2)
        xdxr['datetime'] = pd.to_datetime(xdxr['date_str'])
        xdxr = xdxr.set_index('datetime')
    
        fq = _fetch_xdxr_factor(symbol, adjust, factor_name)
        xdxr = xdxr.join(fq[1:], how='outer')
    return xdxr

def factor_reversion(method: str = 'qfq', raw: pd.DataFrame = None, factor: pd.DataFrame = None) -> pd.DataFrame:
    if factor is not None  and not factor.empty:
        raw.index = pd.to_datetime(raw.index)
        factor.index = pd.to_datetime(factor.index)

        # 按日期升序排列复权因子
        factor = factor.sort_index(ascending=True)
        raw = raw.sort_index(ascending=True)
        # 获取原始数据期间的复权因子
        # 使用最近的可用的复权因子（向后填充）
        factor = factor.reindex(raw.index, method='ffill')

        data = pd.concat([raw, factor.loc[raw.index[0]: raw.index[-1], ['factor']]], axis=1)
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

# 从通达信网站下载日线文件，每日盘前或盘后16:00之后下载，建议盘前
def download_tdx_hsjday() -> io.BytesIO|bool:
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
        print(f'下载文件失败')
        return False


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
        
        if buffer == False:
            return False
        
        buffer.seek(0, io.SEEK_END)
        response_length = buffer.tell()
        filesize = response_length /1024 /1024
        print(f'下载耗时{end-start:.2f}秒, 速度：{filesize/(end-start):.2f} MB/s。')
        if isExtract == True:
            vipdocdir = os.path.join(TDXDIR, 'vipdoc')
            os.makedirs(vipdocdir, exist_ok=True)
            end = datetime.datetime.now().timestamp()

            filenum = 0
            print(f"已下载，文件大小：{filesize:.2f}MB。开始解压文件到: {TDXDIR} 。")
            with zipfile.ZipFile(buffer, 'r') as zip_ref:
                zip_ref.extractall(vipdocdir)
                filenum = len(zip_ref.infolist())
            end2 = datetime.datetime.now().timestamp()
            print(f'解压耗时{end2-end:.2f}秒，解压文件{filenum}个。')

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
        error_msg = f"未知错误: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return False
