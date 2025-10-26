# 进阶配置

最好需要一定的编程基础，自定义组合卖出策略时需要添加代码

## 参数配置

> 修改参数后需要重新启动程序生效

Pool Conf 股票池相关的参数

> * white_indexes 白名单指数列表，买点会选择这些指数的成分券
> * black_prompts 黑名单问财语句，会拉黑当天问财语句里选出来的这些股票绝对不买

Buy Conf 买点相关的参数

> * time_ranges 扫描买点时间段，如果不用买点只用卖点的话可以改成 time_ranges = []
> * interval 扫描买点间隔，取60的约数：1-6, 10, 12, 15, 20, 30
> * order_premium 保证市价单成交的溢价，单位（元）
>
> 其他看代码中详细说明

Sell Conf 卖点相关的参数

> * time_ranges 扫描卖点时间段，如果不用买点只用卖点的话可以改成 time_ranges = []
> * interval 扫描卖点间隔，取60的约数：1-6, 10, 12, 15, 20, 30
> * order_premium 保证市价单成交的溢价，单位（元）
>
> 其他看代码中详细说明

## 注意事项

与手动买卖结合的注意点

> * 尽量保持空仓开始，如果账户预先有股票则可能由于程序未记录持仓历史导致无法正确卖出
> * 确保手动买入的时候程序已经，程序也会自动记录主观买入的持仓
> * 使用过程，需要保证每日开市连续竞价前启动程序，否则无法正确记录持仓时间和历史的最高价导致卖出无法符合预期

本地策略缓存的使用指导

> 要在`CACHE_BASE_PATH`对应的目录里看缓存的信息复盘，可以参考如下：
> * `assets.csv` 里记录的是账户资金曲线历史
> * `deal_hist.csv` 里记录的是交易单委托历史
>
> 要在`CACHE_BASE_PATH`对应的目录里查看缓存是否正确，可以参考如下：
> * `positions.json` 里记录的是持仓天数
> * `max_price.json` 里记录的是历史最高价格

## 组合卖出

可以选择拼接适合固定交易模式的数个卖出组件来构建自定义的`GroupSeller`组合卖出的策略群

以下为预定义的卖出策略单元：

```
Hard Seller: 硬性止损

根据建仓价的下跌比例绝对止损
hard_time_range = ['09:31', '14:57']
earn_limit = 9.999  # 绝对止盈率
risk_limit = 0.979  # 绝对止损率
risk_tight = 0.002  # 换仓下限乘数
```
```
Switch Seller: 换仓卖出

盈利未达预期则卖出换仓
switch_time_range = ['14:30', '14:57']
switch_hold_days = 3             # 持仓天数
switch_require_daily_up = 0.003  # 换仓上限乘数
```
```
Fall Seller: 回落止盈

历史最高价回落比例止盈
fall_time_range = ['09:31', '14:57']
fall_from_top = [
    (1.02, 9.99, 0.02),
    (1.01, 1.02, 0.05),
]
```
```
Return Seller: 回撤止盈

浮盈回撤百分止盈
return_time_range = ['09:31', '14:57']
return_of_profit = [
    (1.07, 9.99, 0.20),
    (1.05, 1.07, 0.50),
    (1.03, 1.05, 0.80),
]
```

```
Open Day Seller: (需要历史数据) 开仓日当天相关参数卖出

opening_time_range = ['14:40', '14:57']
open_low_rate = 0.99     # 低于开仓日最低价比例
open_vol_rate = 0.60     # 低于开仓日成交量比例
```
```
MA Seller: (需要历史数据) 跌破均线卖出

均线一般为价格的一个支撑位
ma_time_range = ['09:31', '14:57']
ma_above = 5  # 跌破N日均线卖出
```
```
CCI Seller: (需要历史数据) CCI 冲高或回落卖出

cci_time_range = ['09:31', '14:57']
cci_upper = 330.0  # cci 高卖点阈值
cci_lower = 10.0   # cci 低卖点阈值
```
```
WR Seller: (需要历史数据) WR上穿卖出

wr_time_range = ['09:31', '14:57']
wr_cross = 25  # wr 卖点阈值
```
```
Volume Drop Seller: (需要历史数据) 次日成交量萎缩卖出

next_time_range = ['09:31', '14:57']
next_volume_dec_threshold = 0.08    # 次日缩量止盈的阈值
next_volume_dec_minute = '09:46'    # 次日缩量止盈的时间点
next_volume_dec_limit = 1.03        # 次日缩量止盈的最大涨幅
```
```
IncidentBlocker: 上涨过程阻断器

开盘一直在上涨的过程中不执行任何卖出（注意GroupSeller的继承顺序）
```
```
Upping Blocker: (需要历史数据) 双涨趋势阻断器

日内均价和MACD同时上升时，不执行后续的卖出策略（注意GroupSeller的继承顺序）
```