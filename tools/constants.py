
MSG_INNER_SEPARATOR = '\n \n'
MSG_OUTER_SEPARATOR = '\n\n '


DEFAULT_DAILY_COLUMNS = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']


# 数据源常量
class DataSource:
    AKSHARE = 'akshare'
    TUSHARE = 'tushare'
    MOOTDX = 'mootdx'
    TDXZIP = 'tdxzip'
    MINIQMT = 'miniqmt'


# 复权常量
class ExitRight:
    BFQ = ''     # 不复权
    QFQ = 'qfq'  # 前复权
    HFQ = 'hfq'  # 后复权


# 逆回购常量
REPURCHASE_CODES = ['131810.SZ', '131811.SZ', '131800.SZ', '131809.SZ', '131801.SZ', '131802.SZ',
                    '131803.SZ', '131805.SZ', '131806.SZ', '204001.SH', '204002.SH', '204003.SH',
                    '204004.SH', '204007.SH', '204014.SH', '204028.SH', '204091.SH', '204182.SH']


# 指数常量
class IndexSymbol:
    INDEX_SH_ZS = '000001'      # 上证指数
    INDEX_SZ_CZ = '399001'      # 深证指数
    INDEX_SZ_50 = '399850'      # 深证50
    INDEX_SZ_100 = '399330'     # 深证100
    INDEX_HS_300 = '000300'     # 沪深300
    INDEX_ZZ_100 = '000903'     # 中证100
    INDEX_ZZ_500 = '000905'     # 中证500
    INDEX_ZZ_800 = '000906'     # 中证800
    INDEX_ZZ_1000 = '000852'    # 中证1000
    INDEX_ZZ_2000 = '932000'    # 中证2000
    INDEX_ZZ_ALL = '000985'     # 中证全指
    INDEX_CY_ZS = '399006'      # 创业指数
    INDEX_KC_50 = '000688'      # 科创50
    INDEX_BZ_50 = '899050'      # 北证50
    INDEX_ZX_100 = '399005'     # 中小100
    INDEX_ZZ_A50 = '000050'     # 中证A50
    INDEX_ZZ_A500 = '000510'    # 中证A500


# 仓位项常量
class InfoItem:
    IncDate = '_inc_date'   # 执行所有持仓日+1操作的日期flag:'%Y-%m-%d'
    DayCount = 'day_count'  # 持仓时间（单位：天）
