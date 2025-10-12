'''
https://www.myquant.cn/docs2/sdk/python/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B.html
需要另建一个虚拟环境，pandas改为1.5.3版本
需要在新环境安装 pip install gm
'''
import logging
import pandas as pd
from gm.api import *

from tools.utils_basic import symbol_to_code, code_to_symbol


# ========================================
# 转换逻辑：dfcf_symbol <-> symbol <-> code
# ========================================
def symbol_to_dfcf_symbol(symbol: str) -> str:
    if symbol[:2] in ['00', '30']:
        return f'SZSE.{symbol}'
    elif symbol[:2] in ['60', '68']:
        return f'SHSE.{symbol}'
    else:
        return symbol + f'BJSE.{symbol}'


def dfcf_symbol_to_symbol(dfcf_symbol: str) -> str:
    arr = dfcf_symbol.split('.')
    assert len(arr) == 2, 'code不符合格式'
    return arr[-1]


def code_to_dfcf_symbol(code: str) -> str:
    return symbol_to_dfcf_symbol(code_to_symbol(code))


def dfcf_symbol_to_code(dfcf_symbol: str) -> str:
    return symbol_to_code(dfcf_symbol_to_symbol(dfcf_symbol))


# ===========
# 回测买卖函数
# ===========
def order_sell(code, price, volume, remark):
    logging.warning(f' {remark} {code}, {price}, {volume}')
    order_volume(
        symbol=code_to_dfcf_symbol(code),
        volume=volume,
        side=OrderSide_Sell,
        order_type=OrderType_Limit,
        position_effect=PositionEffect_Close,
        price=price,
    )


def order_buy(code: str, price: float, volume: int):
    logging.warning(f' 买入: {code}, {price}, {volume}')
    order_volume(
        symbol=code_to_dfcf_symbol(code),
        volume=volume,
        side=OrderSide_Buy,
        order_type=OrderType_Limit,
        position_effect=PositionEffect_Open,
        price=price,
    )


# ===========
# 回测框架函数
# ===========
def get_history_data(context, code: str, days: int, fields: list[str], frequency='1d') -> pd.DataFrame:
    return context.data(
        symbol=code_to_dfcf_symbol(code),
        frequency=frequency,
        count=days,
        fields=','.join(fields),
    )


def update_cache_quote(quotes: dict[str, dict], bars: list[dict], curr_time: str) -> dict[str, dict]:
    # 如果全天无量则说明停牌：删除股票
    if curr_time == '15:00':
        suspensions = set()
        for code in quotes:
            if quotes[code]['lastPrice'] == 0:
                suspensions.add(code)
                print(f'当日停牌 {code} {quotes[code]}')

        for code in suspensions:
            del quotes[code]

    # 报价单改成统一的格式
    for bar in bars:
        code = dfcf_symbol_to_code(bar['symbol'])
        bar['close'] = round(bar['close'], 3)
        bar['open'] = round(bar['open'], 3)
        bar['high'] = round(bar['high'], 3)
        bar['low'] = round(bar['low'], 3)

        if code not in quotes:
            quotes[code] = {'lastClose': None}
            quotes[code]['lastPrice'] = bar['close']
            quotes[code]['open'] = bar['open']
            quotes[code]['high'] = bar['high']
            quotes[code]['low'] = bar['low']
            quotes[code]['volume'] = bar['volume']
            quotes[code]['amount'] = bar['amount']

        if '09:30' < curr_time < '15:00':
            quotes[code]['lastPrice'] = bar['close']
            if quotes[code]['open'] == 0:
                quotes[code]['open'] = bar['open']
            quotes[code]['high'] = max(quotes[code]['high'], bar['high'])
            quotes[code]['low'] = min(quotes[code]['low'], bar['low'])
            quotes[code]['volume'] = quotes[code]['volume'] + bar['volume']
            quotes[code]['amount'] = quotes[code]['amount'] + bar['amount']

        # 如果停牌或者开盘成交量为0则标记为0
        elif curr_time == '15:00':
            quotes[code].clear()
            quotes[code]['lastClose'] = bar['close']
            quotes[code]['lastPrice'] = 0
            quotes[code]['open'] = 0
            quotes[code]['high'] = 0
            quotes[code]['low'] = 0
            quotes[code]['volume'] = 0
            quotes[code]['amount'] = 0

    return quotes


# ===========
# 回测Delegate
# ===========
class EmDelegate:
    """
    主要用东方财富内置的掘金系统做回测，因为可用的数据更多一些
    为了跟GmDelegate作区分，起名EasyMoneyDelegate
    """
    pass
