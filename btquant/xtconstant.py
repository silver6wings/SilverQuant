# -*- coding: utf-8 -*-
"""
xtquant.xtconstant 影子实现。

仅保留 SilverQuant 实际使用到的常量，数值与 xtquant.xtconstant / 迅投数据字典一致。
"""

# 委托类型 order_type（证券买卖）
STOCK_BUY = 23
STOCK_SELL = 24

# 信用账户担保品买卖（与普通买卖同值，xtquant 中独立命名）
CREDIT_BUY = 23
CREDIT_SELL = 24

# 价格类型
FIX_PRICE = 11                 # 限价
LATEST_PRICE = 11              # 最新价
MARKET_PEER_PRICE_FIRST = 14   # 对手方最优价
MARKET_PEER_PRICE_LAST = 15
MARKET_SH_BEST_5_CANCEL = 16
MARKET_SZ_CONVERT_5_CANCEL = 17  # miniqmt：深市市价；大 QMT passorder 映射为 47
MARKET_SH_EDGE_5_CANCEL = 18

# 委托状态 order_status（与内置 Python enum_EEntrustStatus / xtquant.xtconstant 数值一致）
ORDER_UNREPORTED = 48       # 未报
ORDER_WAIT_REPORTING = 49   # 待报
ORDER_REPORTED = 50         # 已报
ORDER_REPORTED_CANCEL = 51  # 已报待撤
ORDER_PARTSUCC_CANCEL = 52  # 部成待撤
ORDER_PART_CANCEL = 53      # 部撤
ORDER_CANCELED = 54         # 已撤
ORDER_PART_SUCC = 55        # 部成
ORDER_SUCCEEDED = 56        # 已成
ORDER_JUNK = 57             # 废单
ORDER_UNKNOWN = 255         # 未知

# 柜台返回的不可撤单 price_type 枚举（xt_delegate.check_orders 过滤用）
BROKER_PRICE_PROP_SUBSCRIBE = 54
BROKER_PRICE_PROP_FUND_ENTRUST = 79
BROKER_PRICE_PROP_ETF = 81
BROKER_PRICE_PROP_DEBT_CONVERSION = 91
