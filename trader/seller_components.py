import logging
from typing import Dict, Optional

import pandas as pd

from mytt.MyTT import MA, MACD, CCI, WR
from xtquant.xttype import XtPosition
from tools.utils_basic import get_limit_up_price
from tools.utils_remote import concat_ak_quote_dict
from trader.seller import BaseSeller


# --------------------------------
# 根据建仓价做硬止损/硬止盈，止损线可随持仓天数上移
# 参数示例：
# hard_time_range = ['09:31', '14:57']
# earn_limit = 1.10
# risk_limit = 0.95
# risk_tight = 0.005
# --------------------------------
class HardSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('硬性卖点模块', end=' ')
        self.hard_time_range = parameters.hard_time_range
        self.earn_limit = parameters.earn_limit
        self.risk_limit = parameters.risk_limit
        self.risk_tight = parameters.risk_tight

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (held_day > 0) and (self.hard_time_range[0] <= curr_time < self.hard_time_range[1]):
            curr_price = quote['lastPrice']
            cost_price = position.open_price
            sell_volume = position.can_use_volume
            switch_lower = cost_price * (self.risk_limit + held_day * self.risk_tight)

            if curr_price <= switch_lower:
                self.order_sell(code, quote, sell_volume, f'跌{int((1 - self.risk_limit) * 100)}%硬止损')
                logging.warning(f'[触发卖出]止损 '
                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} ')
                return True
            elif curr_price >= cost_price * self.earn_limit:
                self.order_sell(code, quote, sell_volume, f'涨{int((self.earn_limit - 1) * 100)}%硬止盈')
                logging.warning(f'[触发卖出]止盈 '
                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} ')
                return True
        return False


# --------------------------------
# 以开盘价/最高价/昨收价中的最大值为参考，回落一定比例即卖出
# 参数示例：
# safe_time_range = ['09:30', '14:57']
# safe_rate = 0.02
# --------------------------------
class SafeSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('走低卖点模块', end=' ')
        self.safe_time_range = parameters.safe_time_range
        self.safe_rate = parameters.safe_rate   # 当日开盘和最高下行百分之多少卖出，例：0.01 = 1% 走低时就开始挂卖单

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (held_day > 0) and (self.safe_time_range[0] <= curr_time < self.safe_time_range[1]):
            curr_price = quote['lastPrice']
            open_price = quote['open']
            high_price = quote['high']
            last_close = quote['lastClose']
            sell_volume = position.can_use_volume

            stop_price = max(last_close, high_price, open_price) * (1 - self.safe_rate)

            if curr_price <= stop_price:
                self.order_sell(code, quote, sell_volume, f'走低{int(self.safe_rate * 100)}%')

                cost_price = position.open_price
                logging.warning(f'[触发卖出]走低 '
                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                    f'开盘价:{round(open_price, 3)} '
                    f'最高价:{round(high_price, 3)} '
                    f'昨收价:{round(last_close, 3)} '
                )
                return True
        return False


# --------------------------------
# 持仓达到指定天数后，若涨幅仍未达到日均目标则换仓卖出
# 参数示例：
# switch_time_range = ['09:35', '14:57']
# switch_hold_days = 2
# switch_demand_daily_up = 0.03
# --------------------------------
class SwitchSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('换仓卖点模块', end=' ')
        self.switch_time_range = parameters.switch_time_range
        self.switch_hold_days = parameters.switch_hold_days
        self.switch_demand_daily_up = parameters.switch_demand_daily_up

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (held_day >= self.switch_hold_days) and (self.switch_time_range[0] <= curr_time < self.switch_time_range[1]):
            curr_price = quote['lastPrice']
            cost_price = position.open_price
            sell_volume = position.can_use_volume
            switch_upper = cost_price * (1 + held_day * self.switch_demand_daily_up)

            if curr_price < switch_upper:  # 未满足盈利目标的仓位
                self.order_sell(code, quote, sell_volume, f'{self.switch_hold_days}日换仓卖单')
                logging.warning(f'[触发卖出]换仓 '
                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} ')
                return True
        return False


# --------------------------------
# 历史最高价回落止盈，按不同涨幅区间配置不同回落阈值
# 参数示例：
# fall_time_range = ['09:40', '14:57']
# fall_from_top = [
#     (1.08, 9.99, 0.05),
#     (1.05, 1.08, 0.04),
#     (1.03, 1.05, 0.03),
# ]
# --------------------------------
class FallSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('回落卖点模块', end=' ')
        self.fall_time_range = parameters.fall_time_range
        self.fall_from_top = parameters.fall_from_top

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if max_price is not None:
            if (held_day > 0) and (self.fall_time_range[0] <= curr_time < self.fall_time_range[1]):
                curr_price = quote['lastPrice']
                cost_price = position.open_price
                sell_volume = position.can_use_volume

                for inc_min, inc_max, fall_threshold in self.fall_from_top:  # 逐级回落卖出
                    if (cost_price * inc_min <= max_price < cost_price * inc_max) \
                            and curr_price < max_price * (1 - fall_threshold):
                        self.order_sell(code, quote, sell_volume,
                                        f'涨{int((inc_min - 1) * 100)}%回落{int(fall_threshold * 100)}%')
                        logging.warning(f'[触发卖出]回落 '
                            f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                            f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                            f'最高:{max_price} 区间:{inc_min}-{inc_max} 回落阈值:{fall_threshold}')
                        return True
        return False


# --------------------------------
# 浮盈回撤止盈，按利润区间要求保留一部分已获得利润
# 参数示例：
# return_time_range = ['09:45', '14:57']
# return_of_profit = [
#     (1.08, 1.15, 0.40),
#     (1.05, 1.08, 0.60),
#     (1.03, 1.05, 0.70),
# ]
# --------------------------------
class ReturnSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('回撤卖点模块', end=' ')
        self.return_time_range = parameters.return_time_range
        self.return_of_profit = parameters.return_of_profit

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if max_price is not None:
            if (held_day > 0) and (self.return_time_range[0] <= curr_time < self.return_time_range[1]):
                curr_price = quote['lastPrice']
                cost_price = position.open_price
                sell_volume = position.can_use_volume

                for inc_min, inc_max, fall_percentage in self.return_of_profit:  # 逐级利润回撤止盈
                    if (cost_price * inc_min <= max_price < cost_price * inc_max) \
                            and curr_price < max_price - (max_price - cost_price) * fall_percentage:
                        self.order_sell(code, quote, sell_volume,
                                        f'涨{int((inc_min - 1) * 100)}%回撤{int(fall_percentage * 100)}%')
                        logging.warning(f'[触发卖出]回撤 '
                            f'成本:{round(cost_price, 3)} 卖价:{round(curr_price, 3)} '
                            f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                            f'最高:{max_price} 区间:{inc_min}-{inc_max} 回撤比值:{fall_percentage}')
                        return True
        return False


# --------------------------------
# 分段移动止盈，达到某段最大涨幅后，回落到保底利润线即卖出
# 例如 (1.03, 1.05, 1.01) 表示最大涨幅在3%~5%之间时，只保留1%利润
# 参数示例：
# move_time_range = ['09:45', '14:57']
# move_profit = [
#     (1.08, 9.99, 1.04),
#     (1.05, 1.08, 1.02),
#     (1.03, 1.05, 1.01),
# ]
# --------------------------------
class MoveSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('移动止盈卖点模块', end=' ')
        self.move_time_range = parameters.move_time_range
        self.move_profit = parameters.move_profit

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if max_price is not None:
            if (held_day > 0) and (self.move_time_range[0] <= curr_time < self.move_time_range[1]):
                curr_price = quote['lastPrice']
                cost_price = position.open_price
                sell_volume = position.can_use_volume

                for inc_min, inc_max, keep_profit in self.move_profit:  # 逐级锁定最低利润
                    if (cost_price * inc_min <= max_price < cost_price * inc_max) \
                            and curr_price <= cost_price * keep_profit:
                        self.order_sell(code, quote, sell_volume,
                                        f'涨{int((inc_min - 1) * 100)}%止盈到{int((keep_profit - 1) * 100)}%')
                        logging.warning(f'[触发卖出]移动止盈 '
                            f'成本:{round(cost_price, 3)} 卖价:{round(curr_price, 3)} '
                            f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                            f'最高:{max_price} 区间:{inc_min}-{inc_max} 保底收益:{keep_profit}')
                        return True
        return False


# --------------------------------
# 开仓日及之后，跌破开仓日低点或尾盘明显缩量时卖出
# 参数示例：
# opening_time_range = ['14:30', '14:57']
# open_low_rate = 1.00
# open_vol_rate = 0.70
# --------------------------------
class OpenDaySeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('开仓日指标止损策略', end=' ')
        self.opening_time_range = parameters.opening_time_range
        self.open_low_rate = parameters.open_low_rate
        self.open_vol_rate = parameters.open_vol_rate

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if history is not None:
            if 0 < held_day < len(history):
                sell_volume = position.can_use_volume
                curr_price = quote['lastPrice']
                cost_price = position.open_price
                open_day_low = history['low'].values[-held_day] * self.open_low_rate

                # 建仓日新低破掉卖
                if curr_price < open_day_low:
                    self.order_sell(code, quote, sell_volume, '破开仓日新低')
                    logging.warning(f'[触发卖出]开仓日最低 '
                        f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                        f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                        f'开仓日最低价:{open_day_low} ')
                    return True

                # 建仓日尾盘缩量卖出
                if curr_price < get_limit_up_price(code, quote['lastClose']):
                    if self.opening_time_range[0] <= curr_time < self.opening_time_range[1]:
                        curr_volume = quote['volume']
                        open_day_volume = history['volume'].values[-held_day] * self.open_vol_rate
                        if curr_volume < open_day_volume:
                            self.order_sell(code, quote, sell_volume, '缩开仓日地量')
                            logging.warning(f'[触发卖出]开仓日缩量 '
                                f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                                f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                                f'建仓日成交量:{open_day_volume}')
                            return True
        return False


# --------------------------------
# 跌破指定均线卖出，适合趋势票破位离场
# 参数示例：
# ma_time_range = ['09:35', '14:57']
# ma_above = 5
# --------------------------------
class MASeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print(f'跌破{parameters.ma_above}日均线卖点模块', end=' ')
        self.ma_time_range = parameters.ma_time_range
        self.ma_above = parameters.ma_above

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if history is not None:
            if (held_day > 0) and (self.ma_time_range[0] <= curr_time < self.ma_time_range[1]):
                sell_volume = position.can_use_volume

                curr_price = quote['lastPrice']

                df = concat_ak_quote_dict(history, quote, curr_date)

                ma_values = MA(df.close.tail(self.ma_above + 1), self.ma_above)
                ma_value = ma_values[-1]

                if curr_price <= ma_value - 0.01:
                    self.order_sell(code, quote, sell_volume, f'破{self.ma_above}日均{ma_value:.2f}')
                    cost_price = position.open_price
                    logging.warning(f'[触发卖出]破均线 '
                        f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                        f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                        f'均线价格:{ma_value}')
                    return True
        return False


# --------------------------------
# 基于 CCI 指标的区间穿越卖出，每5分钟检查一次
# 参数示例：
# cci_time_range = ['09:35', '14:57']
# cci_upper = 100
# cci_lower = -100
# --------------------------------
class CCISeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('CCI卖点模块', end=' ')
        self.cci_time_range = parameters.cci_time_range
        self.cci_upper = parameters.cci_upper
        self.cci_lower = parameters.cci_lower

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (history is not None) and (self.cci_time_range[0] <= curr_time < self.cci_time_range[1]):
            if (held_day > 0) and int(curr_time[-2:]) % 5 == 0:  # 每隔5分钟 CCI 卖出
                sell_volume = position.can_use_volume

                df = concat_ak_quote_dict(history, quote, curr_date)

                df['CCI'] = CCI(df['close'], df['high'], df['low'], 14)
                cci = df['CCI'].tail(2).values

                if cci[0] > self.cci_lower > cci[1]:  # CCI 下穿
                    self.order_sell(code, quote, sell_volume, f'CCI高于{self.cci_lower}')

                    curr_price = quote['lastPrice']
                    cost_price = position.open_price
                    logging.warning(f'[触发卖出]CCI '
                        f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                        f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                        f'当前CCI下沿:{self.cci_lower}')
                    return True

                if cci[0] < self.cci_upper < cci[1]:  # CCI 上穿
                    self.order_sell(code, quote, sell_volume, f'CCI低于{self.cci_upper}')
                    curr_price = quote['lastPrice']
                    cost_price = position.open_price
                    logging.warning(f'[触发卖出]CCI '
                        f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                        f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                        f'当前CCI上沿:{self.cci_upper}')
                    return True
        return False


# --------------------------------
# WR 指标上穿阈值卖出，每5分钟检查一次
# 参数示例：
# wr_time_range = ['09:35', '14:57']
# wr_cross = 80
# --------------------------------
class WRSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('WR上穿卖点模块', end=' ')
        self.wr_time_range = parameters.wr_time_range
        self.wr_cross = parameters.wr_cross

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (history is not None) and (self.wr_time_range[0] <= curr_time < self.wr_time_range[1]):
            if held_day > 0 and int(curr_time[-2:]) % 5 == 0:  # 每隔5分钟 WR 卖出
                sell_volume = position.can_use_volume

                df = concat_ak_quote_dict(history, quote, curr_date)

                df['WR'] = WR(df['close'], df['high'], df['low'], 14)
                wr = df['WR'].tail(2).values

                if wr[0] < self.wr_cross < wr[1]:  # WR 上穿
                    self.order_sell(code, quote, sell_volume, f'WR上穿{self.wr_cross}卖')
                    curr_price = quote['lastPrice']
                    cost_price = position.open_price
                    logging.warning(f'[触发卖出]WR '
                                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                                    f'当前WR阈值:{self.wr_cross}')

                    return True
        return False


# --------------------------------
# 次日固定时刻检查缩量，若缩量且仍有利润则卖出
# 参数示例：
# next_time_range = ['09:30', '10:30']
# vol_dec_thre = 0.60
# vol_dec_time = '10:00'
# vol_dec_limit = 1.095
# --------------------------------
class VolumeDropSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('次缩卖点模块', end=' ')
        self.next_time_range = parameters.next_time_range
        self.next_volume_dec_threshold = parameters.vol_dec_thre
        self.next_volume_dec_minute = parameters.vol_dec_time
        self.next_volume_dec_limit = parameters.vol_dec_limit

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (history is not None) and (self.next_time_range[0] <= curr_time < self.next_time_range[1]):
            cost_price = position.open_price
            sell_volume = position.can_use_volume

            prev_close = quote['lastClose']
            curr_price = quote['lastPrice']
            curr_vol = quote['volume']

            # 次缩止盈：开盘至今成交量相比买入当日总成交量，缩量达标盈利则卖出，除非涨停
            if held_day > 0 and curr_time == self.next_volume_dec_minute:
                open_vol = history['volume'].values[-held_day]
                if curr_vol < open_vol * self.next_volume_dec_threshold \
                        and cost_price < curr_price < prev_close * self.next_volume_dec_limit:
                    self.order_sell(code, quote, sell_volume, '次日缩量')
                    logging.warning(f'[触发卖出]次缩 '
                                    f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                                    f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                                    f'次缩阈值:{self.next_volume_dec_threshold}')
                    return True
        return False


# --------------------------------
# 高开后快速走弱卖出，按高开幅度分段配置允许的回落阈值
# 参数示例：
# drop_time_range = ['09:31', '10:30']
# drop_out_limits = [
#     (1.03, 1.05, 0.02),
#     (1.05, 1.07, 0.025),
#     (1.07, 9.99, 0.03),
# ]
# --------------------------------
class DropSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('高开出货卖点模块', end=' ')
        self.drop_time_range = parameters.drop_time_range
        self.drop_out_limits = parameters.drop_out_limits

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if (held_day > 0) and (self.drop_time_range[0] <= curr_time < self.drop_time_range[1]):
            sell_volume = position.can_use_volume

            if quote['lastPrice'] < quote['open']:
                opn = round(quote['open'], 3)
                low = round(quote['low'], 3)
                hgh = round(quote['high'], 3)
                clz = round(quote['lastPrice'], 3)

                if clz == low and (hgh - opn < opn - clz):  # 下跌过程且实心大于上影线
                    last_close = quote['lastClose']
                    open_price = opn
                    drop_price = opn - clz

                    for inc_min, inc_max, drop_threshold in self.drop_out_limits:  # 逐级高开卖出
                        if last_close * inc_min <= open_price < last_close * inc_max \
                                and drop_price > last_close * drop_threshold:
                            self.order_sell(code, quote, sell_volume,
                                            f'高开{int((inc_min - 1) * 100)}跌{int(drop_threshold * 100)}%')

                            curr_price = quote['lastPrice']
                            cost_price = position.open_price
                            logging.warning(f'[触发卖出]高开 '
                                f'成本:{round(cost_price, 3)} 现价:{round(curr_price, 3)} '
                                f'涨跌:{round((curr_price / cost_price - 1) * 100, 3)} '
                                f'高开范围:{inc_min}-{inc_max} 下落:{drop_threshold}%')
                            return True

        return False


# --------------------------------
# 单根强势上涨形态阻断器，命中后返回 True 以阻止其他 Seller 卖出
# 参数示例：无需额外参数
# --------------------------------
class IncBlocker(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('上涨过程禁卖模块', end=' ')

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if held_day > 0:
            if quote['lastPrice'] > quote['open'] \
                    and round(quote['high'], 3) == round(quote['lastPrice'], 3) \
                    and round(quote['open'], 3) == round(quote['low'], 3):
                return True
        return False


# --------------------------------
# 趋势上行阻断器，MACD 与价格同步上行时阻止其他 Seller 卖出
# 参数示例：无需额外参数
# --------------------------------
class UppingBlocker(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('上行趋势禁卖模块', end=' ')

    def check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        if history is not None:
            if held_day > 0:
                df = concat_ak_quote_dict(history, quote, curr_date)

                _, _, df['MACD'] = MACD(df['close'])
                macd = df['MACD'].tail(2).values

                close = df['close'].tail(2).values
                high = df['high'].tail(2).values
                low = df['low'].tail(2).values

                yesterday_price = close[0] + high[0] + low[0]
                today_price = close[1] + high[1] + low[1]

                if macd[0] < macd[1] and yesterday_price < today_price:  # macd上行 & 价格上行
                    # self.order_sell(code, quote, sell_volume, '上行不卖')
                    return True
        return False
