import math
import datetime
import logging

from delegate.base_delegate import BaseDelegate

from tools.utils_basic import get_limit_up_price, debug, is_stock_kc


DEFAULT_BUY_REMARK = '买入委托'
PENDING_CONFIRM_TICKS = 3           # 下单后等待持仓确认的轮数（如每秒一轮，则约 3 秒）
MARKET_ORDER_PRICE_RATE = 1.017     # 市价单保护价：现价 101.7%，留余量应对 tick 延迟与 2% 价格笼子


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
        self.order_status: dict[str, int] = {}   # 市价单待确认买入：code -> 等待轮数
        self._order_status_date: str = ''
        self.update_config(parameters)

    def update_config(self, parameters) -> None:
        """
        允许动态更新更新设置
        """
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
        available_cash: float = 0.0,
        all_in_buy: bool = False,   # 最后一点零头不够也要尝试买入
        all_market: bool = True,    # 全部都是市价单
    ) -> dict[str, set]:
        if curr_date not in today_buy:
            today_buy[curr_date] = set()

        if all_market:
            self._sync_order_status_date(curr_date)
            self._tick_pending_orders(today_buy, curr_date, positions)

        if len(selections) > 0:
            final_capacity = self.slot_capacity

            position_codes = [position.stock_code for position in positions]
            position_count = self.delegate.get_holding_position_count(positions)
            if available_cash <= 0.0:
                available_cash = self.delegate.check_asset().cash
            available_slot = available_cash // final_capacity

            # 不足一手把剩下的钱尽可能买一手
            if available_slot == 0 and all_in_buy:
                final_capacity = available_cash - 1.00
                available_slot = 1

            pending_count = len(self.order_status) if all_market else 0
            available_slot = min(
                available_slot,
                self.daily_buy_max - len(today_buy[curr_date]) - pending_count,
            )

            buy_count = max(0, self.slot_count - position_count)    # 确认剩余的仓位
            buy_count = min(buy_count, available_slot)              # 确认现金够用
            buy_count = min(buy_count, len(selections))             # 确认选出的股票够用
            buy_count = min(buy_count, self.once_buy_limit)         # 限制一秒内下单数量
            buy_count = int(buy_count)

            for code in selections:  # 依次买入
                if buy_count > 0:
                    if code in today_buy[curr_date]:
                        continue
                    if all_market and code in self.order_status:
                        continue

                    selection = selections[code]
                    price = round(selection[SelectionItem.BUY_PRICE], 4)
                    last_close = round(selection[SelectionItem.LAST_CLOSE], 4)

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
                        if self.order_buy(
                            code=code,
                            price=price,
                            last_close=last_close,
                            volume=buy_volume,
                            remark=remark,
                            market=all_market,
                        ):
                            if all_market:
                                self.order_status[code] = 0
                                logging.warning(f'[待确认买入]{code}\t现价: {price:.2f}')
                            elif code not in today_buy[curr_date]:
                                today_buy[curr_date].add(code)
                                logging.warning(f"[记录选股]{code}\t现价: {price:.2f}")
                else:
                    break
        return today_buy

    def _sync_order_status_date(self, curr_date: str) -> None:
        if self._order_status_date != curr_date:
            self.order_status.clear()
            self._order_status_date = curr_date

    @staticmethod
    def _is_today_locked_position(code: str, positions: list) -> bool:
        for position in positions:
            if position.stock_code == code:
                return position.volume > 0 and position.can_use_volume == 0
        return False

    def _tick_pending_orders(
        self,
        today_buy: dict[str, set],
        curr_date: str,
        positions: list,
    ) -> None:
        for code in list(self.order_status.keys()):
            self.order_status[code] += 1
            if self.order_status[code] < PENDING_CONFIRM_TICKS:
                continue

            if self._is_today_locked_position(code, positions):
                del self.order_status[code]
                today_buy[curr_date].add(code)
                logging.warning(f'[确认买入]{code} 持仓已锁定(can_use_volume=0)，记入当日买入')
            else:
                del self.order_status[code]
                logging.warning(f'[待确认买入]{code} 超时未确认，允许重试')

    def order_buy(
        self,
        code: str,
        price: float,
        last_close: float,
        volume: int,
        remark: str,
        market: bool = True,
        log: bool = True,
    ) -> bool:
        buy_volume = volume
        if self.risk_control and buy_volume > self.slot_capacity / price:
            buy_volume = math.floor(self.slot_capacity / price / 100) * 100
            logging.warning(f'[触发风控]{code} 超过风险控制，买入量调整为 {buy_volume} 股')

        if buy_volume < 1:
            logging.warning(f'[取消委托]{code} 挂单买量=0')
            return False

        if buy_volume < 200 and is_stock_kc(code):
            logging.warning(f'[取消委托]{code} 科创最少200')
            return False

        limit_price = get_limit_up_price(code, last_close)
        if market:
            # 按比例取保护价，避免固定 premium 低价股超 2% 价格笼子、高价股溢价不足
            order_price = min(round(price * MARKET_ORDER_PRICE_RATE, 2), limit_price)
        else:
            order_price = min(round(price + self.order_premium, 2), limit_price)

        if market:
            buy_type = '市买'
            if order_price >= limit_price and limit_price > 0:
                # 接近涨停：保护价触顶，改挂涨停限价单
                final_price = limit_price
                self.delegate.order_limit_open(
                    code=code,
                    price=final_price,
                    volume=buy_volume,
                    remark=remark,
                    strategy_name=self.strategy_name)
            else:
                final_price = order_price
                self.delegate.order_market_open(
                    code=code,
                    price=final_price,
                    volume=buy_volume,
                    remark=remark,
                    strategy_name=self.strategy_name)
        else:
            buy_type = '限买'
            final_price = order_price
            self.delegate.order_limit_open(
                code=code,
                price=final_price,
                volume=buy_volume,
                remark=remark,
                strategy_name=self.strategy_name)

        if log:
            logging.warning(f'[{buy_type}委托]{code}\t委托价:{final_price:.3f} {buy_volume}股 {remark} ')

        if self.delegate.callback is not None:
            self.delegate.callback.record_order(
                order_time=datetime.datetime.now().timestamp(),
                code=code,
                price=price,
                volume=buy_volume,
                side=f'{buy_type}委托',
                remark=remark)

        return True


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
