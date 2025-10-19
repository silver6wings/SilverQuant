import datetime
import logging

from decimal import Decimal, ROUND_HALF_UP


IS_DEBUG = False


def debug(*args, **kwargs):
    if IS_DEBUG:
        print(*args, **kwargs)


# pandas dataframe 显示配置优化
def pd_show_all() -> None:
    import pandas as pd
    pd.set_option('display.width', None)
    pd.set_option('display.min_rows', 9999)
    pd.set_option('display.max_rows', 9999)
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)
    pd.set_option('display.float_format', lambda x: f'{x:.3f}')


# logging 模块的初始化配置
def logging_init(path=None, level=logging.DEBUG, file_line=False):
    file_line_fmt = ""
    if file_line:
        file_line_fmt = "%(filename)s[line:%(lineno)d] - %(levelname)s: "
    logging.basicConfig(
        level=level,
        format=file_line_fmt + "%(asctime)s|%(message)s",
        filename=path
    )


# 多文件 logger的配置
def logger_init(path=None, name='default') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 移除已存在的处理器
    for handler in logger.handlers:
        logger.removeHandler(handler)

    if path is None:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
    else:
        handler = logging.FileHandler(path)
        handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s|%(message)s')
        handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


# 六位数symbol代码转换成带交易所后缀code格式
def symbol_to_code(symbol: str | int) -> str:
    symbol = str(symbol) if type(symbol) == int else symbol

    if symbol[:2] in ['00', '30', '15', '12']:
        return f'{symbol}.SZ'
    elif symbol[:2] in ['60', '68', '51', '52', '53', '56', '58', '11']:
        return f'{symbol}.SH'
    elif symbol[:2] in ['83', '87', '43', '82', '88', '92']:
        return f'{symbol}.BJ'
    else:
        return f'{symbol}.--'


# 带交易所后缀code格式转换成六位数symbol代码
def code_to_symbol(code: str) -> str:
    arr = code.split('.')
    assert len(arr) == 2, 'code不符合格式'
    return arr[0]


# ==========
# 通达信系列代码
# ==========


def code_to_tdxsymbol(code: str) -> str:
    [symbol, exchange] = code.split('.')
    if exchange == 'SZ':
        return '0' + symbol
    elif exchange == 'SH':
        return '1' + symbol
    elif exchange == 'BJ':
        return '2' + symbol
    return code         # 这里先不变，不报错


def tdxsymbol_to_code(tdxsymbol: str) -> str:
    if len(tdxsymbol) != 7:
        return tdxsymbol    # 这里先不变，不报错
    elif tdxsymbol[0] == '0':
        return tdxsymbol[1:7] + '.SZ'
    elif tdxsymbol[0] == '1':
        return tdxsymbol[1:7] + '.SH'
    elif tdxsymbol[0] == '2':
        return tdxsymbol[1:7] + '.BJ'
    else:
        return tdxsymbol    # 这里先不变，不报错


def symbol_to_tdxsymbol(code: str) -> str:
    """ 转换代码函数 """
    # 深证为0，沪市为1，北交所为2
    code = ''.join(c for c in code if c.isdigit())  # 只取股票代码中数字代码部分
    # A股，股票代码转换，如：1601068，2300250
    if len(code) == 6:
        if code[0] == "6" or code[0] == "9":  # 上证股票
            return "1" + code
        if code[0] == "0" or code[0] == "3" or code[0] == "2":  # 深证股票
            return "0" + code
        if code[0] == "4" or code[0] == "8":  # 北证股票
            return "2" + code
    return code


# ==========
# 掘金系列代码
# ==========


def symbol_to_gmsymbol(symbol: str | int) -> str:
    symbol = str(symbol) if type(symbol) == int else symbol

    if symbol[:2] in ['00', '30', '15', '12']:
        return f'SZSE.{symbol}'
    elif symbol[:2] in ['60', '68', '51', '52', '53', '56', '58', '11']:
        return f'SHSE.{symbol}'
    elif symbol[:2] in ['83', '87', '43', '82', '88', '92']:
        return f'BJSE.{symbol}'
    else:
        return f'--SE.{symbol}'


def gmsymbol_to_symbol(gmsymbol: str) -> str:
    arr = gmsymbol.split('.')
    assert len(arr) == 2, 'code不符合格式'
    return arr[-1]


def code_to_gmsymbol(code: str) -> str:
    return symbol_to_gmsymbol(code_to_symbol(code))


def gmsymbol_to_code(gmsymbol: str) -> str:
    return symbol_to_code(gmsymbol_to_symbol(gmsymbol))


# 判断是不是可交易股票代码 包含 股票 ETF 可转债
def is_symbol(code_or_symbol: str):
    return code_or_symbol[:2] in [
        '00', '30',  # 深交所
        '60', '68',  # 上交所
        '82', '83', '87', '88', '43', '92',  # 北交所
        '15', '51', '52', '53', '56', '58',  # ETF
        '11', '12',  # 可转债
    ]


def is_stock(code_or_symbol: str | int):
    """ 判断是不是股票代码 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['00', '30', '60', '68', '82', '83', '87', '88', '43', '92']


def is_stock_10cm(code_or_symbol: str | int):
    """ 判断是不是10cm票 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['00', '60']


def is_stock_20cm(code_or_symbol: str | int):
    """ 判断是不是20cm票 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['30', '68']


def is_stock_30cm(code_or_symbol: str | int):
    """ 判断是不是20cm票 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['82', '83', '87', '88', '43', '92']


def is_stock_cy(code_or_symbol: str | int):
    """ 判断是不是创业板 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] == '30'


def is_stock_kc(code_or_symbol: str | int):
    """ 判断是不是科创板 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] == '68'


def is_stock_bj(code_or_symbol: str | int):
    """ 判断是不是北交所 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['82', '83', '87', '88', '43', '92']


def is_fund_etf(code_or_symbol: str | int):
    """ 判断是不是etf代码 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['15', '51', '52', '53', '56', '58']


def is_bond(code_or_symbol: str | int):
    """ 判断是不是可转债 """
    code_or_symbol = str(code_or_symbol) if type(code_or_symbol) == int else code_or_symbol
    return code_or_symbol[:2] in ['11', '12']


# 获取symbol的交易所简称
def get_symbol_exchange(symbol: str) -> str:
    if symbol[:2] in ['00', '30', '15', '12']:
        return 'SZ'
    elif symbol[:2] in ['60', '68', '51', '52', '53', '56', '58', '11']:
        return 'SH'
    elif symbol[:2] in ['83', '87', '43', '82', '88', '92']:
        return 'BJ'
    else:
        return ''


# 获取code的交易所简称
def get_code_exchange(code: str) -> str:
    arr = code.split('.')
    assert len(arr) == 2, 'code不符合格式'
    return arr[1][:2]


# 大数字转换成字母码
def map_num_to_chr(num):
    quotient = num // 100
    if quotient < 10:
        return chr(quotient + 48)  # 将数字转换为对应的 ASCII 字符
    elif quotient < 36:
        return chr(quotient - 10 + 97)  # 将数字转换为小写字母
    elif quotient < 62:
        return chr(quotient - 36 + 65)  # 将数字转换为大写字母
    else:
        return '.'


# 获取当前时间在一天连续竞价交易时间的百分位
def get_current_time_percentage(time: str) -> float:
    [hr, mn, sc] = time.split(':')
    if hr == '09' and '30' <= mn <= '59':
        tsc = ((int(hr) - 9) * 60 + int(mn) - 30) * 60 + int(sc)
    elif hr == '10':
        tsc = ((int(hr) - 9) * 60 + int(mn) - 30) * 60 + int(sc)
    elif hr == '11' and '0' <= mn <= '30':
        tsc = ((int(hr) - 9) * 60 + int(mn) - 30) * 60 + int(sc)
    elif '13' <= hr <= '15':
        tsc = ((int(hr) - 13 + 2) * 60 + int(mn)) * 60 + int(sc)
    else:
        return -1

    return float(tsc) / 3600 / 4


# 获取涨停幅
def get_limiting_up_rate(code_or_symbol: str) -> float:
    if code_or_symbol[:2] == '30' or code_or_symbol[:2] == '68':
        return 1.2
    elif code_or_symbol[:1] == '8' or code_or_symbol[:1] == '9' or code_or_symbol[:1] == '4':
        return 1.3
    else:
        return 1.1


# 获取涨停价
def get_limit_up_price(code_or_symbol: str, pre_close: float) -> float:
    if pre_close is None or pre_close <= 0:
        return 0.0

    limit_rate = get_limiting_up_rate(code_or_symbol)

    pre_close_dec = Decimal(str(pre_close)).quantize(Decimal('0.00'))
    limit_price_dec = pre_close_dec * Decimal(str(limit_rate))
    limit_price_dec = limit_price_dec.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)

    return float(limit_price_dec)


# 获取跌停幅
def get_limiting_down_rate(code_or_symbol: str) -> float:
    if code_or_symbol[:2] == '30' or code_or_symbol[:2] == '68':
        return 0.8
    elif code_or_symbol[:1] == '8' or code_or_symbol[:1] == '9' or code_or_symbol[:1] == '4':
        return 0.7
    else:
        return 0.9


# 获取跌停价
def get_limit_down_price(code_or_symbol: str, pre_close: float) -> float:
    if pre_close is None or pre_close <= 0:
        return 0.0

    limit_rate = get_limiting_down_rate(code_or_symbol)

    pre_close_dec = Decimal(str(pre_close)).quantize(Decimal('0.00'))
    limit_price_dec = pre_close_dec * Decimal(str(limit_rate))
    limit_price_dec = limit_price_dec.quantize(Decimal('0.00'), rounding=ROUND_HALF_UP)

    return float(limit_price_dec)


# TODO
def get_st_limit_up_rate():
    return


# TODO
def get_st_limit_up_price():
    return


# TODO
def get_st_limit_down_rate():
    return


# TODO
def get_st_limit_down_price():
    return


def time_diff_seconds(later_time: datetime.datetime.time, early_time: datetime.datetime.time):
    """ 将时间转换为总秒数 """
    total_seconds_time1 = later_time.hour * 3600 + later_time.minute * 60 + later_time.second
    total_seconds_time2 = early_time.hour * 3600 + early_time.minute * 60 + early_time.second

    # 计算两个时间之间的秒数差
    diff_seconds = abs(total_seconds_time1 - total_seconds_time2)

    return diff_seconds


# 迅投相关数据
# past_seconds 当日交易日已经过去多少秒，计算量比这类指标需要
def hms_to_past_seconds(hour: int, minute: int, second: int) -> int:
    time_int = hour * 100 + minute
    if time_int < 930:
        return 0
    elif 930 <= time_int < 1130:
        return hour * 3600 + minute * 60 + second - (3600 * 9 + 60 * 30)
    elif 1130 <= time_int < 1300:
        return 3600 * 2
    elif 1300 <= time_int < 1500:
        return hour * 3600 + minute * 60 + second - (3600 * 13) + 3600 * 2
    else:
        return 3600 * 4


def xt_time_tag_to_hms(time_tag: str) -> tuple[int, int, int]:
    dt = datetime.datetime.strptime(time_tag, '%Y%m%d %H:%M:%S')
    return dt.hour, dt.minute, dt.second


def xt_time_tag_to_past_seconds(time_tag: int | str) -> int:
    hour, minute, second = xt_time_tag_to_hms(str(time_tag))
    return hms_to_past_seconds(hour, minute, second)


# sub_whole 的 quotes 时间格式
def xt_time_to_hms(timestamp: int) -> tuple[int, int, int]:
    if len(str(timestamp)) > 10:
        timestamp = timestamp // 1000
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.hour, dt.minute, dt.second


def xt_time_to_past_seconds(timestamp: int) -> int:
    hour, minute, second = xt_time_to_hms(timestamp)
    return hms_to_past_seconds(hour, minute, second)
