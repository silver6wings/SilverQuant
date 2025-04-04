import time
import datetime

import pywencai
import akshare as ak

from mytt.MyTT_advance import *
from tools.utils_basic import symbol_to_code


# （过时）筛选东方财富行业板块的公式
def select_industry_sections(
    df: pd.DataFrame,
    fp: int = 10,
    sp: int = 22,
    ap: int = 7,
) -> bool:
    C = df.close

    _, _, df['MACD'] = MACD(
        C,
        SHORT=fp,
        LONG=sp,
        M=ap,
    )
    df['SLOPE'] = SLOPE(df['MACD'], 5)

    df['AA'] = (df['SLOPE'] > 0) & (df['MACD'] > 0)

    df['BB'] = EMA(C, 5) > EMA(C, 10)

    df['CC'] = SLOPE(EMA(C, 5), 3) > 0

    df['SAFE'] = df['AA'] & df['BB'] & df['CC']

    return df['SAFE'].values[-1]


# （过时）筛选东方财富行业板块的逻辑
def select_dfcf_industry_sections(
    section_names: list[str],
    start_date: str = None,
    end_date: str = None,
    adjust: str = 'qfq',
):
    # 根据指标筛选板块
    now = datetime.datetime.now()
    if start_date is None:
        start_date = (now - datetime.timedelta(days=50)).strftime("%Y%m%d")
    if end_date is None:
        end_date = (now - datetime.timedelta(days=1)).strftime("%Y%m%d")

    section_result = []
    for section_name in section_names:
        print(section_name, end=' ')

        time.sleep(1)
        df = ak.stock_board_industry_hist_em(
            symbol=section_name,
            start_date=start_date,
            end_date=end_date,
            period="日k",
            adjust=adjust,
        ).rename(columns={
            '日期': 'datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
        })[['datetime', 'open', 'close', 'high', 'low', 'volume', 'amount']]

        if select_industry_sections(df):
            section_result.append(section_name)

    print(f'\n{len(section_result)}/{len(section_names)}')
    print(section_result)

    return section_result


# 选择东方财富的行业板块逻辑
def get_dfcf_industry_sections(limit: int = 2000) -> list[str]:
    # 初筛板块
    df = ak.stock_board_industry_name_em()
    df['涨跌比'] = (df['上涨家数'] + 1) / (df['下跌家数'] + 1)
    df = df.sort_values(by=['涨跌比'], ascending=False)

    df['总家数'] = df['上涨家数'] + df['下跌家数']

    df = df.drop(df[df['上涨家数'] / df['下跌家数'] <= 1.0].index)
    # print(df)

    total_sum = 0
    count = 0
    for value in df["总家数"]:
        total_sum += value
        count += 1
        if total_sum > limit:
            break

    section_names = df.head(count)['板块名称'].values
    return section_names


# 获取东方财富行业板块成份
def get_dfcf_industry_stock_codes(section_result: list[str]) -> set:
    stock_list = set()
    for section_name in section_result:
        df = ak.stock_board_industry_cons_em(symbol=section_name)
        codes = {symbol_to_code(symbol) for symbol in df['代码'].values}
        stock_list.update(codes)

    return stock_list


# 选择同花顺概念板块的逻辑
def get_ths_concept_sections(limit: int = 2000, period: int = 0):
    assert period in {0, 3, 5, 10, 20}, '{"即时", "3日排行", "5日排行", "10日排行", "20日排行"}'

    symbol = '即时' if period == 0 else f'{period}日排行'

    df = ak.stock_fund_flow_concept(symbol=symbol)
    # print(df)

    df = df.drop(df[df['净额'] < 0].index)
    if symbol == '即时':
        up_rate = '行业-涨跌幅'
    else:
        up_rate = '阶段涨跌幅'
        df[up_rate] = df[up_rate].str.rstrip('%').astype(float)

    df = df.drop(df[df[up_rate] < 0].index)
    df['涨净比'] = df[up_rate] / df['净额']
    df = df.sort_values(by='涨净比', ascending=False)

    total_sum = 0
    count = 0
    for value in df['公司家数']:
        total_sum += value
        count += 1
        if total_sum > limit:
            break

    df = df.head(count)
    # print(df)

    section_names = df['行业']
    return section_names.values


# 获取同花顺概念板块成份
def get_ths_concept_stock_codes(section_names: list[str]):
    stock_list = set()
    for section_name in section_names:
        query = f'{section_name}概念板块'
        df = pywencai.get(query=query, perpage=100, loop=True)
        if df is not None and type(df) != dict and df.shape[0] > 0:
            codes = df['股票代码'].values
            stock_list.update(codes)

    return stock_list


# 获取申万指数列表
def get_sw_sections(target_date='20240809'):
    df = ak.index_analysis_daily_sw(
        symbol="二级行业",
        start_date=target_date,
        end_date=target_date,
    )
    df = df.rename(columns={
        '指数代码': 'symbol',
        '指数名称': 'name',
        '发布日期': 'datetime',
        '收盘指数': 'close',
        '成交量': 'volume',
        '涨跌幅': 'up_rate',
        '换手率': 'switch_rate',
        '市盈率': 'P/E',
        '市净率': 'P/B',
        '均价': 'price_avg',
        '成交额占比': 'amount_pct',
        '流通市值': 'market_value',
        '平均流通市值': 'market_value_avg',
        '股息率': 'dividend_rate',
    })
    df = df.sort_values('switch_rate', ascending=False)
    print(df)
