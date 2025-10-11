# coding=utf-8
# from __future__ import print_function, absolute_import

import sys
import math
from typing import List, Dict, Set, Optional

import numpy as np
from mytt.MyTT_advance import SMA, WMA, ATR
from tools.utils_cache import get_prefixes_stock_codes
from tools.utils_dfcf import *


_FILENAME = sys.argv[0].split('\\')[-1]
_BACKTEST_INTERVAL = '300s'
_BACKTEST_START_TIME = '2025-04-12 09:30:00'
_BACKTEST_END_TIME = '2025-04-30 15:00:00'
_BACKTEST_INIT_CASH = 100000

code_list = get_prefixes_stock_codes({'00', '60'}, True)

cache_quotes: Dict[str, Dict] = {}          # 记录实时的价格信息
cache_select: Dict[str, Set] = {}           # 记录选股历史，去重
cache_indicators: Dict[str, Dict] = {}      # 记录技术指标相关值

held_days = {}


class Parameter:
    # 下单持仓
    switch_begin = '09:45'  # 每天最早换仓时间
    hold_days = 3           # 持仓天数
    max_count = 10          # 持股数量上限
    amount_each = 10000     # 每个仓的资金上限
    order_premium = 0.08    # 保证成功下单成交的溢价
    upper_buy_count = 3     # 单次选股最多买入股票数量（若单次未买进当日不会再买这只
    # 止盈止损
    upper_income = 1.45     # 止盈率（ATR失效时使用）
    lower_income = 0.97     # 止损率（ATR失效时使用）
    sw_upper_multi = 0.02   # 换仓上限乘数
    sw_lower_multi = 0.005  # 换仓下限乘数
    atr_time_period = 3     # 计算atr的天数
    atr_upper_multi = 1.25  # 止盈atr的乘数
    atr_lower_multi = 0.85  # 止损atr的乘数
    sma_time_period = 3     # 卖点sma的天数
    # 策略参数
    L = 20                  # 选股HMA短周期
    M = 40                  # 选股HMA中周期
    N = 60                  # 选股HMA长周期
    S = 5                   # 选股SMA周期
    inc_limit = 1.02        # 相对于昨日收盘的涨幅限制
    min_price = 2.00        # 限制最低可买入股票的现价
    # 历史指标
    day_count = 69          # 70个足够算出周期为60的 HMA
    data_cols = ['close', 'high', 'low']    # 历史数据需要的列


# ======== 盘前 ========


def calculate_indicators(data: pd.DataFrame) -> Optional[Dict]:
    row_close = data['close']
    row_high = data['high']
    row_low = data['low']

    close_3d = row_close.tail(Parameter.atr_time_period).values
    high_3d = row_high.tail(Parameter.atr_time_period).values
    low_3d = row_low.tail(Parameter.atr_time_period).values

    return {
        'PAST_69': row_close.tail(Parameter.day_count).values,
        'CLOSE_3D': close_3d,
        'HIGH_3D': high_3d,
        'LOW_3D': low_3d,
    }


def before_trade_day(context) -> None:
    for code in held_days:
        held_days[code] += 1

    history_codes = code_list
    count = 0
    for code in history_codes:
        temp_df = get_history_data(context, code, Parameter.day_count, ['close', 'high', 'low'])
        if not temp_df.isnull().values.any() and len(temp_df) == Parameter.day_count:
            cache_indicators[code] = calculate_indicators(temp_df)
            count += 1
    print(f'{count} stock indicators prepared')


# ======== 买点 ========


def get_hma(data: np.array, n: int) -> np.array:
    wma1 = WMA(data, n // 2)
    wma2 = WMA(data, n)
    sqrt_n = int(np.sqrt(n))

    diff = 2 * wma1 - wma2
    hma = WMA(diff, sqrt_n)

    return hma


def get_sma(data: np.array, n: int) -> np.array:
    sma = SMA(data, n)
    return sma


def decide_stock(quote: Dict, indicator: Dict) -> (bool, Dict):
    curr_close = quote['lastPrice']
    curr_open = quote['open']
    last_close = quote['lastClose']

    if not curr_close > Parameter.min_price:
        return False, {}

    if not curr_open < curr_close < last_close * Parameter.inc_limit:
        return False, {}

    sma05 = get_sma(np.append(indicator['PAST_69'], [curr_close]), Parameter.S)
    sma60 = get_sma(np.append(indicator['PAST_69'], [curr_close]), Parameter.L)

    if not (sma05[-2] < sma05[-1]):
        return False, {}

    if not (sma60[-2] < sma60[-1] < last_close):
        return False, {}

    hma60 = get_hma(np.append(indicator['PAST_69'], [curr_close]), Parameter.N)
    hma40 = get_hma(np.append(indicator['PAST_69'], [curr_close]), Parameter.M)
    hma20 = get_hma(np.append(indicator['PAST_69'], [curr_close]), Parameter.L)

    if not (hma20[-1] < hma40[-1] < hma60[-1]):
        return False, {}

    if hma20[-2] < hma40[-2] < hma60[-2]:
        return False, {}

    return True, {}


def select_stocks(quotes: Dict) -> List[Dict[str, any]]:
    selections = []
    for code in quotes:
        if code not in cache_indicators:
            continue

        passed, info = decide_stock(quotes[code], cache_indicators[code])
        if passed:
            selection = {'code': code, 'price': quotes[code]['lastPrice']}
            selection.update(info)
            selections.append(selection)
    return selections


def scan_buy(context, quotes, curr_date, curr_time):
    selections = select_stocks(quotes)
    print(selections, end='')

    # 选出一个以上的股票
    if len(selections) > 0:
        selections = sorted(selections, key=lambda x: x['price'])  # 选出的股票按照现价从小到大排序

        position_codes = [
            dfcf_symbol_to_code(position['symbol'])
            for position in context.account().positions()
        ]
        position_count = len(position_codes)
        available_cash = context.account().cash['available']

        buy_count = max(0, Parameter.max_count - position_count)            # 确认剩余的仓位
        buy_count = min(buy_count, available_cash // Parameter.amount_each)     # 确认现金够用
        buy_count = min(buy_count, len(selections))                 # 确认选出的股票够用
        buy_count = min(buy_count, Parameter.upper_buy_count)               # 限制一秒内下单数量
        buy_count = int(buy_count)

        for i in range(len(selections)):  # 依次买入
            if buy_count > 0:
                code = selections[i]['code']
                price = selections[i]['price']
                buy_volume = math.floor(Parameter.amount_each / price / 100) * 100

                if buy_volume <= 0:
                    logging.debug(f'{code} 价格过高')
                elif code in position_codes:
                    logging.debug(f'{code} 正在持仓')
                elif curr_date in cache_select and code in cache_select[curr_date]:
                    logging.debug(f'{code} 今日已选')
                else:
                    buy_count = buy_count - 1

                    # 如果今天未被选股过 and 目前没有持仓则记录（意味着不会加仓
                    order_buy(code, price, buy_volume)
                    held_days[code] = 0
            else:
                break


# ======== 卖点 ========


def get_last_1_sma(row_close, period) -> float:
    sma = SMA(row_close, period)
    return sma[-1]


def get_atr(row_close, row_high, row_low, period) -> float:
    atr = ATR(row_close, row_high, row_low, period)
    return atr[-1]


def scan_sell(context, quotes, curr_date, curr_time):
    for position in context.account().positions():
        code = dfcf_symbol_to_code(position['symbol'])
        if (code in quotes) and (code in held_days):
            # 如果有数据且有持仓时间记录
            quote = quotes[code]
            curr_price = quote['lastPrice']
            cost_price = position['vwap']
            sell_volume = position['volume']

            # 换仓：未满足盈利目标的仓位
            held_day = held_days[code]
            switch_upper = cost_price * (1 + held_day * Parameter.sw_upper_multi)
            switch_lower = cost_price * (Parameter.lower_income + held_day * Parameter.sw_lower_multi)

            if held_day > Parameter.hold_days and curr_time >= Parameter.switch_begin:
                if switch_lower < curr_price < switch_upper:
                    order_sell(code, curr_price, sell_volume, '换仓卖单')

            # 判断持仓超过一天
            if held_day > 0:
                if (code in quotes) and (code in cache_indicators):
                    if curr_price <= switch_lower:
                        # 绝对止损卖出
                        order_sell(code, curr_price, sell_volume, 'ABS止损委托')
                    elif curr_price >= cost_price * Parameter.upper_income:
                        # 绝对止盈卖出
                        order_sell(code, curr_price, sell_volume, 'ABS止盈委托')
                    else:
                        quote = quotes[code]
                        close = np.append(cache_indicators[code]['CLOSE_3D'], quote['lastPrice'])
                        high = np.append(cache_indicators[code]['HIGH_3D'], quote['high'])
                        low = np.append(cache_indicators[code]['LOW_3D'], quote['low'])

                        sma = get_last_1_sma(close, Parameter.sma_time_period)
                        atr = get_atr(close, high, low, Parameter.atr_time_period)

                        atr_upper = sma + atr * Parameter.atr_upper_multi
                        atr_lower = sma - atr * Parameter.atr_lower_multi

                        if curr_price <= atr_lower:
                            # ATR止损卖出
                            order_sell(code, curr_price, sell_volume, 'ATR止损委托')
                        elif curr_price >= atr_upper:
                            # ATR止盈卖出
                            order_sell(code, curr_price, sell_volume, 'ATR止盈委托')
                else:
                    if curr_price <= switch_lower:
                        # 默认绝对止损卖出
                        order_sell(code, curr_price, sell_volume, 'DEF止损委托')
                    elif curr_price >= cost_price * Parameter.upper_income:
                        # 默认绝对止盈卖出
                        order_sell(code, curr_price, sell_volume, 'DEF止盈委托')


# ==== 框架部分 ====


def execute_strategy(context, curr_date: str, curr_time: str, prev_time: str, quotes: dict):
    # 盘前
    if prev_time == '09:30':
        before_trade_day(context)
        print(f'========== {curr_date} START ==========')

    print(curr_date, curr_time, len(quotes), end=':')
    # 早盘
    if '09:31' <= curr_time <= '11:30':
        scan_sell(context, quotes, curr_date, curr_time)
        scan_buy(context, quotes, curr_date, curr_time)
    # 午盘
    elif '13:00' <= curr_time <= '14:56':
        scan_sell(context, quotes, curr_date, curr_time)

    print('')


def init(context):
    context.backtest_interval = _BACKTEST_INTERVAL
    context.period = Parameter.day_count
    context.symbol = [code_to_dfcf_symbol(code) for code in code_list]

    # 订阅行情
    subscribe(symbols=context.symbol, frequency=_BACKTEST_INTERVAL, wait_group=True)
    subscribe(symbols=context.symbol, frequency='1d', count=context.period)


def on_bar(context, bars):
    if bars[0]['frequency'] != context.backtest_interval:
        return

    now = bars[0]['eob']
    curr_date = now.strftime('%Y-%m-%d')
    curr_time = now.strftime('%H:%M')
    prev_time = bars[0]['bob'].strftime('%H:%M')

    global cache_quotes
    cache_quotes = update_cache_quote(cache_quotes, bars, curr_time)

    # 执行预定策略
    if '09:30' < curr_time < '14:57':
        legal_quotes = {
            k: v
            for k, v in cache_quotes.items()
            if v['lastClose'] is not None and v['lastPrice'] != 0
        }
        if len(legal_quotes) > 0:
            execute_strategy(context, curr_date, curr_time, prev_time, legal_quotes)


if __name__ == '__main__':
    '''
        strategy_id策略ID, 由系统生成
        filename文件名, 请与本文件名保持一致
        mode运行模式, 实时模式:MODE_LIVE回测模式:MODE_BACKTEST
        token绑定计算机的ID, 可在系统设置-密钥管理中生成
        backtest_start_time回测开始时间
        backtest_end_time回测结束时间
        backtest_adjust股票复权方式, 不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
        backtest_initial_cash回测初始资金
        backtest_commission_ratio回测佣金比例
        backtest_slippage_ratio回测滑点比例
        backtest_match_mode市价撮合模式，以下一tick/bar开盘价撮合:0，以当前tick/bar收盘价撮合：1
        '''
    run(strategy_id='94928c8f-a42a-11f0-99d4-c6b1e26dfd7a',
        filename=_FILENAME,
        mode=MODE_BACKTEST,
        token='970a008e02b631efd0d7fa53caeed0ee700e2342',
        backtest_start_time=_BACKTEST_START_TIME,
        backtest_end_time=_BACKTEST_END_TIME,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=_BACKTEST_INIT_CASH,
        backtest_commission_ratio=0.0005,  # 万五的手续费
        backtest_slippage_ratio=0.0001,  # 万一的滑点
        backtest_match_mode=1)
