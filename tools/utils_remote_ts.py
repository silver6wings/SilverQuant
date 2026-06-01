import time
from typing import Optional

import pandas as pd

from tools.constants import DEFAULT_DAILY_COLUMNS, ExitRight
from tools.utils_basic import is_stock


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
TUSHARE_BATCH_RETRY_TIMES = 3
TUSHARE_BATCH_MIN_SPLIT_SIZE = 10
TUSHARE_BATCH_RETRY_DEPTH_LIMIT = 12


def _is_tushare_rate_limit_error(exception: Exception) -> bool:
    message = str(exception)
    return '频率超限' in message or 'rate limit' in message.lower()


def get_ts_daily_history(
    code: str,
    start_date: str,  # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> Optional[pd.DataFrame]:
    # 当前使用 tushare daily 接口，只返回不复权数据；保留 adjust 参数兼容旧调用。
    if not is_stock(code):
        return None

    from reader.tushare_agent import get_tushare_pro
    try_times = 0
    df = None
    last_exception = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1
        try:
            pro = get_tushare_pro()
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        except Exception as e:
            last_exception = e
            df = None
            if _is_tushare_rate_limit_error(e):
                break
            if try_times < 3:
                time.sleep(0.5)

    if last_exception is not None and df is None:
        print(f'[TUSHARE] skip {code} {start_date}-{end_date}: {last_exception}')
        return None

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


# 复合版:通过返回dict的key区分不同的票，注意总共一次最多6000行会限制长度
# https://tushare.pro/document/2?doc_id=27
def _get_ts_daily_histories_by_batch(
    codes: list[str],
    start_date: str,
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    retry_depth: int = 0,
    interval: float = 1,
) -> dict[str, pd.DataFrame]:
    if len(codes) == 0:
        return {}

    from reader.tushare_agent import get_tushare_pro

    if retry_depth > TUSHARE_BATCH_RETRY_DEPTH_LIMIT:
        print(f'[TUSHARE] skip {len(codes)} codes {start_date}-{end_date}: retry depth exceeded')
        return {}

    try_times = 0
    df = None
    last_exception = None
    while (df is None or len(df) <= 0) and try_times < TUSHARE_BATCH_RETRY_TIMES:
        try_times += 1
        try:
            pro = get_tushare_pro()
            df = pro.daily(ts_code=','.join(codes), start_date=start_date, end_date=end_date)
            time.sleep(interval)
        except Exception as e:
            last_exception = e
            df = None
            if _is_tushare_rate_limit_error(e):
                break
            if try_times < TUSHARE_BATCH_RETRY_TIMES:
                time.sleep(interval)

    if last_exception is not None and df is None:
        if _is_tushare_rate_limit_error(last_exception):
            print(f'[TUSHARE] skip {len(codes)} codes {start_date}-{end_date}: rate limit, stop retry: {last_exception}')
            return {}

        if len(codes) <= TUSHARE_BATCH_MIN_SPLIT_SIZE:
            print(f'[TUSHARE] skip {len(codes)} codes {start_date}-{end_date}: failed after {TUSHARE_BATCH_RETRY_TIMES} tries: {last_exception}')
            return {}

        mid = len(codes) // 2
        left_codes = codes[:mid]
        right_codes = codes[mid:]
        print(f'[TUSHARE] batch failed {len(codes)} codes {start_date}-{end_date}, split retry: {last_exception}')

        ans = {}
        ans.update(_get_ts_daily_histories_by_batch(left_codes, start_date, end_date, columns, retry_depth + 1))
        ans.update(_get_ts_daily_histories_by_batch(right_codes, start_date, end_date, columns, retry_depth + 1))
        return ans

    ans = {}
    if df is not None and len(df) > 0:
        if 'ts_code' not in df.columns:
            if len(codes) <= TUSHARE_BATCH_MIN_SPLIT_SIZE:
                print(f'[TUSHARE] skip {len(codes)} codes {start_date}-{end_date}: missing ts_code column')
                return {}

            mid = len(codes) // 2
            left_codes = codes[:mid]
            right_codes = codes[mid:]
            print(f'[TUSHARE] batch malformed {len(codes)} codes {start_date}-{end_date}, split retry: missing ts_code column')

            ans.update(_get_ts_daily_histories_by_batch(left_codes, start_date, end_date, columns, retry_depth + 1))
            ans.update(_get_ts_daily_histories_by_batch(right_codes, start_date, end_date, columns, retry_depth + 1))
            return ans

        returned_codes = set(df['ts_code'].dropna().unique())
        for code in codes:
            if code in returned_codes:
                temp_df = df[df['ts_code'] == code]
                temp_df = _ts_to_standard(temp_df)

                if columns is None:
                    ans[code] = temp_df
                else:
                    ans[code] = temp_df[columns]

        if len(returned_codes) == 0:
            if len(codes) <= TUSHARE_BATCH_MIN_SPLIT_SIZE:
                print(f'[TUSHARE] skip {len(codes)} codes {start_date}-{end_date}: no matched ts_code returned')
                return {}

            mid = len(codes) // 2
            left_codes = codes[:mid]
            right_codes = codes[mid:]
            print(f'[TUSHARE] batch unmatched {len(codes)} codes {start_date}-{end_date}, split retry')

            ans.update(_get_ts_daily_histories_by_batch(left_codes, start_date, end_date, columns, retry_depth + 1))
            ans.update(_get_ts_daily_histories_by_batch(right_codes, start_date, end_date, columns, retry_depth + 1))
            return ans

        missing_codes = [code for code in codes if code not in returned_codes]
        if len(missing_codes) > 0:
            if len(missing_codes) >= len(codes):
                print(f'[TUSHARE] skip retry {len(codes)} codes {start_date}-{end_date}: no progress from partial result')
                return ans
            if len(missing_codes) <= TUSHARE_BATCH_MIN_SPLIT_SIZE:
                print(f'[TUSHARE] skip retry missing {len(missing_codes)} codes {start_date}-{end_date}: below retry split size')
                return ans
            print(f'[TUSHARE] partial result {len(returned_codes)}/{len(codes)} codes {start_date}-{end_date}, retry missing {len(missing_codes)} codes')
            ans.update(_get_ts_daily_histories_by_batch(missing_codes, start_date, end_date, columns, retry_depth + 1))
    return ans


def get_ts_daily_histories(
    codes: list[str],
    start_date: str,    # format: 20240101
    end_date: str,
    columns: list[str] = DEFAULT_DAILY_COLUMNS,
    adjust: ExitRight = ExitRight.BFQ,
) -> dict[str, pd.DataFrame]:
    # 当前使用 tushare daily 接口，只返回不复权数据；保留 adjust 参数兼容旧调用。
    for code in codes:
        if not is_stock(code):
            print(f'存在不符合格式要求的code: {code}')
            return {}

    return _get_ts_daily_histories_by_batch(codes, start_date, end_date, columns)
