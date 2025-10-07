import math
import datetime
import logging

from delegate.base_delegate import BaseDelegate

from tools.utils_basic import get_limit_up_price, debug


DEFAULT_BUY_REMARK = '买入委托'


class SelectionItem:
    BUY_PRICE = 'price'
    BUY_VOLUME = 'volume'
    LAST_CLOSE = 'lastClose'    # 昨日收盘价主要用来判断涨跌停


class BaseBuyer:
    def __init__(
        self,
        strategy_name: str,
        delegate: BaseDelegate,
        parameters,     # Buy Configuration
    ):
        self.strategy_name = strategy_name
        self.delegate = delegate

        self.order_premium = parameters.order_premium
        self.slot_capacity = parameters.slot_capacity
        self.slot_count = parameters.slot_count
        self.daily_buy_max = parameters.daily_buy_max
        self.once_buy_limit = parameters.once_buy_limit

        self.risk_control = parameters.risk_control if hasattr(parameters, 'risk_control') else False

    def buy_selections(
        self,
        selections: dict[str, dict],    # { code: quote } 注意 Python 3.7 之前的dict不按照插入序遍历
        today_buy: dict[str, set],      # 当日已买入记录
        curr_date: str,
        positions: list,
        remark: str = DEFAULT_BUY_REMARK,
        all_in_buy: bool = False,   # 最后一点零头不够也要尝试买入
        all_market: bool = True,    # 全部都是市价单
    ) -> dict[str, set]:
        if len(selections) > 0:
            final_capacity = self.slot_capacity

            position_codes = [position.stock_code for position in positions]
            position_count = self.delegate.get_holding_position_count(positions)
            available_cash = self.delegate.check_asset().cash
            available_slot = available_cash // final_capacity

            # 不足一手把剩下的钱尽可能买一手
            if available_slot == 0 and all_in_buy:
                final_capacity = available_cash - 1.00
                available_slot = 1

            # 每日最多买入限制要有
            if curr_date not in today_buy:
                today_buy[curr_date] = set()
            available_slot = min(available_slot, self.daily_buy_max - len(today_buy[curr_date]))

            buy_count = max(0, self.slot_count - position_count)    # 确认剩余的仓位
            buy_count = min(buy_count, available_slot)              # 确认现金够用
            buy_count = min(buy_count, len(selections))             # 确认选出的股票够用
            buy_count = min(buy_count, self.once_buy_limit)         # 限制一秒内下单数量
            buy_count = int(buy_count)

            for code in selections:  # 依次买入
                if buy_count > 0:
                    if code in today_buy[curr_date]:
                        continue

                    selection = selections[code]
                    price = round(selection[SelectionItem.BUY_PRICE], 2)
                    last_close = round(selection[SelectionItem.LAST_CLOSE], 2)

                    if SelectionItem.BUY_VOLUME in selection:
                        buy_volume = selection[SelectionItem.BUY_VOLUME]
                    else:
                        buy_volume = math.floor(final_capacity / price / 100) * 100

                    if buy_volume <= 0:
                        debug(f'[{code} 不够一手]')
                    elif code in position_codes:
                        debug(f'[{code} 正在持仓]')
                    else:
                        buy_count = buy_count - 1
                        # 如果今天未被选股过 and 目前没有持仓则记录（意味着不会加仓
                        self.order_buy(
                            code=code, price=price, last_close=last_close,
                            volume=buy_volume, remark=remark, market=all_market)
                        # 记录买入历史
                        if code not in today_buy[curr_date]:
                            today_buy[curr_date].add(code)
                            logging.warning(f"记录选股 {code}\t现价: {price:.2f}")
                else:
                    break
        return today_buy

    def order_buy(
        self,
        code: str,
        price: float,
        last_close: float,
        volume: int,
        remark: str,
        market: bool = True,
        log: bool = True,
    ):
        buy_volume = volume
        if self.risk_control and buy_volume > self.slot_capacity / price:
            buy_volume = math.floor(self.slot_capacity / price / 100) * 100
            logging.warning(f'{code} 超过风险控制，买入量调整为 {buy_volume} 股')

        if buy_volume > 0:
            order_price = price + self.order_premium
            limit_price = get_limit_up_price(code, last_close)

            if market:
                buy_type = '市买'
                if order_price > limit_price:
                    # 如果涨停了只能挂限价单
                    self.delegate.order_limit_open(
                        code=code,
                        price=limit_price,
                        volume=buy_volume,
                        remark=remark,
                        strategy_name=self.strategy_name)
                else:
                    self.delegate.order_market_open(
                        code=code,
                        price=min(order_price, limit_price),
                        volume=buy_volume,
                        remark=remark,
                        strategy_name=self.strategy_name)
            else:
                buy_type = '限买'
                self.delegate.order_limit_open(
                    code=code,
                    price=min(order_price, limit_price),
                    volume=buy_volume,
                    remark=remark,
                    strategy_name=self.strategy_name)

            if log:
                logging.warning(f'{buy_type}委托 {code} \t现价:{price:.3f} {buy_volume}股')

            if self.delegate.callback is not None:
                self.delegate.callback.record_order(
                    order_time=datetime.datetime.now().timestamp(),
                    code=code,
                    price=price,
                    volume=buy_volume,
                    side=f'{buy_type}委托',
                    remark=remark)
        else:
            print(f'{code} 挂单买量为0，不委托')


class LimitedBuyer(BaseBuyer):
    def __init__(
        self,
        strategy_name: str,
        delegate: BaseDelegate,
        parameters,
        volume_ratio: float = 1.00,  # 每次下单的 volume 是 capacity 的百分比可以调整
    ):
        super().__init__(
            strategy_name,
            delegate,
            parameters,
        )
        self.volume_ratio = volume_ratio

    def order_buy(
        self,
        code: str,
        price: float,
        last_close: float,
        volume: int,
        remark: str,
        market: bool = True,
        log: bool = True,
    ):
        volume = math.floor(volume / 100 * self.volume_ratio) * 100     # 向下取整
        super().order_buy(code, price, last_close, volume, remark, market, log)
