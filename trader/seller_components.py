import logging

from mytt.MyTT_advance import *
# from mytt.MyTT_custom import *
from typing import Dict, Optional

from xtquant.xttype import XtPosition
from tools.utils_basic import get_limit_up_price
from tools.utils_remote import concat_ak_quote_dict
from trader.seller import BaseSeller


# --------------------------------
# 根据建仓价的下跌比例严格绝对止损
# --------------------------------
class HardSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('硬性卖出策略', end=' ')
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
                return True
            elif curr_price >= cost_price * self.earn_limit:
                self.order_sell(code, quote, sell_volume, f'涨{int((self.earn_limit - 1) * 100)}%硬止盈')
                return True
        return False


# --------------------------------
# 盈利未达预期则卖出换仓
# --------------------------------
class SwitchSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('换仓卖出策略', end=' ')
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
                return True
        return False


# --------------------------------
# 历史最高价回落比例止盈
# --------------------------------
class FallSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('回落卖出策略', end=' ')
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
                        logging.warning(f'[Sell]'
                                        f'cost_p:{cost_price} max_p:{max_price} '
                                        f'inc_min:{inc_min} inc_max:{inc_max}')
                        self.order_sell(code, quote, sell_volume,
                                        f'涨{int((inc_min - 1) * 100)}%回落{int(fall_threshold * 100)}%')
                        return True
        return False


# --------------------------------
# 浮盈回撤百分止盈
# --------------------------------
class ReturnSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('回撤卖出策略', end=' ')
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

                for inc_min, inc_max, fall_percentage in self.return_of_profit:  # 逐级回落止盈
                    if (cost_price * inc_min <= max_price < cost_price * inc_max) \
                            and curr_price < max_price - (max_price - cost_price) * fall_percentage:
                        logging.warning(f'[Sell]'
                                        f'cost_p:{cost_price} max_p:{max_price} '
                                        f'inc_min:{inc_min} inc_max:{inc_max}')
                        self.order_sell(code, quote, sell_volume,
                                        f'涨{int((inc_min - 1) * 100)}%回撤{int(fall_percentage * 100)}%')
                        return True
        return False


# # --------------------------------
# # 尾盘涨停卖出（暂时先别用）
# # --------------------------------
# class TailCapSeller(BaseSeller):
#     def __init__(self, strategy_name, delegate, parameters):
#         BaseSeller.__init__(self, strategy_name, delegate, parameters)
#         print('尾盘涨停卖出策略', end=' ')
#         self.tail_time_range = parameters.tail_time_range
#
#     def check_sell(self, code: str, quote: Dict, curr_date: str, curr_time: str, position: XtPosition,
#                    held_day: int, max_price: Optional[float], history: Optional[pd.DataFrame]) -> bool:
#
#         if history is not None:
#             if (held_day > 0) and (self.tail_time_range[0] <= curr_time < self.tail_time_range[1]):
#                 sell_volume = position.can_use_volume
#                 curr_price = quote['lastPrice']
#                 last_close = history['close'].values[-1]
#
#                 # TODO: 似乎有点bug，怎么判断是尾盘涨停？
#                 if curr_price >= get_limit_up_price(code, last_close):
#                     self.order_sell(code, quote, sell_volume, '尾盘涨停')
#                     return True
#         return False


# --------------------------------
# 开仓日当天指标尾盘止损
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
                open_day_low = history['low'].values[-held_day] * self.open_low_rate

                # 建仓日新低破掉卖
                if curr_price < open_day_low:
                    self.order_sell(code, quote, sell_volume, '破开仓日新低')
                    return True

                # 建仓日尾盘缩量卖出
                if curr_price < get_limit_up_price(code, quote['lastClose']):
                    if self.opening_time_range[0] <= curr_time < self.opening_time_range[1]:
                        curr_volume = quote['volume']
                        open_day_volume = history['volume'].values[-held_day] * self.open_vol_rate
                        if curr_volume < open_day_volume:
                            self.order_sell(code, quote, sell_volume, '缩开仓日地量')
                            return True
        return False


# --------------------------------
# 跌破均线卖出
# --------------------------------
class MASeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print(f'跌破{parameters.ma_above}日均线卖出策略', end=' ')
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
                    return True
        return False


# --------------------------------
# CCI 冲高回落卖出
# --------------------------------
class CCISeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('CCI卖出策略', end=' ')
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
                    return True

                if cci[0] < self.cci_upper < cci[1]:  # CCI 上穿
                    self.order_sell(code, quote, sell_volume, f'CCI低于{self.cci_upper}')
                    return True
        return False


# --------------------------------
# WR上穿卖出
# --------------------------------
class WRSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('WR上穿卖出策略', end=' ')
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
                    return True
        return False


# --------------------------------
# 次日成交量萎缩卖出
# --------------------------------
class VolumeDropSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('次缩卖出策略', end=' ')
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
                    return True
        return False


# --------------------------------
# 高开出货卖出
# --------------------------------
class DropSeller(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('高开出货卖出', end=' ')
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
                            return True

        return False


# --------------------------------
# 上涨过程阻断器
# --------------------------------
class IncBlocker(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('上涨过程禁卖', end=' ')

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
# 双涨趋势阻断器
# --------------------------------
class UppingBlocker(BaseSeller):
    def __init__(self, strategy_name, delegate, parameters):
        BaseSeller.__init__(self, strategy_name, delegate, parameters)
        print('上行趋势禁卖', end=' ')

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
