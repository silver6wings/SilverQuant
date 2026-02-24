import datetime
import akshare as ak

from mytt.MyTT_advance import *


# 指数MA均线指标择时
def get_ma_index_indicator(
    symbol: str = '000985',
    period: int = 5,
) -> tuple[bool, dict]:
    end_dt = datetime.datetime.now() - datetime.timedelta(days=0)
    start_dt = end_dt - datetime.timedelta(days=250)  # EMA 时间必须够长
    df = ak.index_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_dt.strftime('%Y%m%d'),
        end_date=end_dt.strftime('%Y%m%d'),
    )
    close = df['收盘'].values
    df['MA5'] = MA(close, period)
    df['SAFE'] = df['MA5'] < df['收盘']
    return df['SAFE'].values[-1], {'df': df}


# 指数MACD指标择时
def get_macd_index_indicator(
    symbol: str = '000985',
    fp: int = 12,
    sp: int = 26,
    ap: int = 9,
    sa: int = 5,  # 斜率周期
) -> tuple[bool, dict]:
    end_dt = datetime.datetime.now() - datetime.timedelta(days=0)
    start_dt = end_dt - datetime.timedelta(days=250)  # EMA 时间必须够长
    df = ak.index_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_dt.strftime('%Y%m%d'),
        end_date=end_dt.strftime('%Y%m%d'),
    )
    close = df['收盘'].values

    # DIF = EMA(CLOSE, 10) - EMA(CLOSE, 22)
    # DEA = EMA(DIF, 7)
    # macd = (DIF - DEA) * 2
    # macd = MACD(CLOSE, 10, 22, 7)
    # print(macd)

    _, _, df['MACD'] = MACD(
        close,
        SHORT=fp,
        LONG=sp,
        M=ap,
    )
    df['SLOPE'] = SLOPE(df['MACD'], sa)
    df['SAFE'] = (df['MACD'] > 0) & (df['SLOPE'] > 0) | (df['SLOPE'] > 8)  # MACD和斜率同时 > 0 或者 斜率大于8

    return df['SAFE'].values[-1], {'df': df}
