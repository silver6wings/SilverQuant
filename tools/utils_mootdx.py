import os
import time
import datetime
import pandas as pd


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


def get_xdxr(symbol: str, cache_dir: str = DEFAULT_XDXR_CACHE_PATH, expire_hours: int = 12):
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

        if xdxr_data is not None:  # 简单判断数据有效性
            xdxr_data.to_csv(cache_file, index=False)  # index=False不保存索引列

        return xdxr_data
    except Exception as e:
        print(f' mootdx get xdxr {symbol} error: ', e)
        return None


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
