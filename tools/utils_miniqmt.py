import threading
import pandas as pd

from xtquant import xtdata
from tools.constants import *


def _run_with_timeout(target_func, args=(), timeout=1) -> any:
    """
    在子线程中执行目标函数，超时未返回则抛出TimeoutException

    :param target_func: 要执行的目标函数
    :param args: 目标函数的参数（元组形式）
    :param timeout: 超时时间（秒）
    :return: 目标函数的返回值
    """
    # 用列表存储结果（因为列表是可变对象，子线程可修改）
    result_container = [None]

    # 子线程执行的包装函数（负责调用目标函数并保存结果）
    def thread_wrapper():
        result_container[0] = target_func(*args)

    # 创建并启动子线程
    thread = threading.Thread(target=thread_wrapper)
    thread.start()

    # 主线程等待子线程结束，最多等待timeout秒
    thread.join(timeout)

    # 如果子线程仍在运行，说明超时
    if thread.is_alive():
        raise Exception(f"函数执行超过 {timeout} 秒，触发超时")

    # 否则返回结果
    return result_container[0]


def _download_and_fetch_qmt_daily(code_list: list[str], start_time: str, end_time: str, adjust: ExitRight) -> dict:
    xtdata.enable_hello = False
    period = '1d'

    # 大规模download容易莫名卡死，老问题了
    downloaded_code_list = []
    for code in code_list:
        try:
            _run_with_timeout(
                target_func=xtdata.download_history_data,
                args=(code, period, start_time, end_time, None),
                timeout=3,
            )
            downloaded_code_list.append(code)
        except Exception as e:
            print(f'{code}:下载{e}')

    # 除权类型 "none" "front" "back" "front_ratio" "back_ratio"
    if adjust == ExitRight.QFQ:
        dividend_type = 'front'
    elif adjust == ExitRight.QFQ:
        dividend_type = 'back'
    else:
        dividend_type = 'none'

    data = xtdata.get_market_data(
        field_list=['time', 'open', 'close', 'high', 'low', 'volume', 'amount'],
        # "time"  # 时间戳
        # "open"  # 开盘价
        # "high"  # 最高价
        # "low"  # 最低价
        # "close"  # 收盘价
        # "volume"  # 成交量
        # "amount"  # 成交额
        # "settle"  # 今结算
        # "openInterest"  # 持仓量
        stock_list=downloaded_code_list,
        period=period,
        start_time=start_time,
        end_time=end_time,
        count=-1,
        dividend_type=dividend_type,
        fill_data=False,
    )
    return data


def _qmt_to_standard(input_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    processed_dfs = []

    for col_name, df in input_dict.items():
        transposed = df.transpose()
        transposed = transposed.reset_index().rename(columns={'index': 'date_str'})
        long_format = transposed.melt(
            id_vars=['date_str'],
            var_name='code',
            value_name=col_name
        )
        processed_dfs.append(long_format)

    merged_df = processed_dfs[0]
    for df in processed_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on=['date_str', 'code'], how='outer')

    merged_df['datetime'] = merged_df['date_str'].astype(int)
    merged_df = merged_df.drop('date_str', axis=1)
    merged_df = merged_df.rename(columns={'time': 'timestamp'})
    merged_df[['open', 'close', 'high', 'low']] = merged_df[['open', 'close', 'high', 'low']].round(2)
    merged_df['amount'] = merged_df['amount'].round(2)

    column_order = ['code', 'datetime'] + [
        col for col in merged_df.columns
        if col not in ['code', 'datetime']
    ]
    merged_df = merged_df[column_order]

    return merged_df


def get_qmt_daily_histories(
    code_list: list[str],
    start_time: str,
    end_time: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ,
) -> pd.DataFrame:
    data = _download_and_fetch_qmt_daily(code_list, start_time, end_time, adjust)

    df = _qmt_to_standard(data)
    if df is not None and len(df) > 0:
        if columns is not None:
            return df[columns]
        return df
    return None


def get_qmt_daily_history(
    code: str,
    start_time: str,
    end_time: str,
    columns: list[str] = None,
    adjust: ExitRight = ExitRight.BFQ
) -> pd.DataFrame:
    return get_qmt_daily_histories([code], start_time, end_time, columns, adjust)


# if __name__ == '__main__':
#     pd_show_all()
#
#     codes = ['600000.SH', '601919.SH']
#     start = '20200601'
#     end = '20200601'
#
#     df = get_qmt_daily_histories(codes, start, end, None, ExitRight.QFQ)
#     print(df)
#
#     df = get_qmt_daily_history(codes[0], start, end, None, ExitRight.QFQ)
#     print(df)
