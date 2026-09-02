# -*- coding: utf-8 -*-
"""
xtquant.xtconstant 影子实现。

仅保留SilverQuant实际使用到的常量。真实 xtquant 里这些常量的整数值会随版本变化，
这里取业界最常见且与 buyer.py 里 `order.order_type == 1`（买）判断一致的取值：
STOCK_BUY=1 / STOCK_SELL=2。
"""
# 订单方向
STOCK_BUY = 1
STOCK_SELL = 2

# 信用交易方向（部分版本使用，预留兼容）
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

# 委托状态（仅保留回调判断常用的几个）
ORDER_UNKNOWN = 48
ORDER_REPORTED = 50
ORDER_REPORTED_CANCEL = 51
ORDER_PARTSUCC_CANCEL = 52
ORDER_PART_CANCEL = 53
ORDER_SUCC = 55
ORDER_CANCEL = 56
ORDER_REJECTED = 57
ORDER_SUSPENDED = 60
ORDER_DELETED = 255

# 柜台返回的不可撤单 price_type 枚举（xt_delegate.check_orders 过滤用）
BROKER_PRICE_PROP_SUBSCRIBE = 54
BROKER_PRICE_PROP_FUND_ENTRUST = 79
BROKER_PRICE_PROP_ETF = 81
BROKER_PRICE_PROP_DEBT_CONVERSION = 91
