import datetime
import logging
import pandas as pd
from typing import List, Dict, Optional

from xtquant.xttype import XtPosition

from delegate.base_delegate import BaseDelegate
from tools.utils_basic import get_limit_down_price


class BaseSeller:
    def __init__(self, strategy_name: str, delegate: BaseDelegate, parameters):
        self.strategy_name = strategy_name
        self.delegate = delegate
        self.order_premium = parameters.order_premium

    def order_sell(self, code, quote, volume, remark, log=True) -> None:
        if volume > 0:
            order_price = quote['lastPrice'] - self.order_premium
            limit_price = get_limit_down_price(code, quote['lastClose'])
            if order_price < limit_price:
                # 如果跌停了只能挂限价单
                self.delegate.order_limit_close(
                    code=code,
                    price=limit_price,
                    volume=volume,
                    remark=remark,
                    strategy_name=self.strategy_name)
            else:
                self.delegate.order_market_close(
                    code=code,
                    price=order_price,
                    volume=volume,
                    remark=remark,
                    strategy_name=self.strategy_name)

            if log:
                logging.warning(f'{remark} {code}\t现价:{order_price:.3f} {volume}股')

            if self.delegate.callback is not None:
                self.delegate.callback.record_order(
                    order_time=datetime.datetime.now().timestamp(),
                    code=code,
                    price=order_price,
                    volume=volume,
                    side='卖出委托',
                    remark=remark)

        else:
            print(f'{code} 挂单卖量为0，不委托')

    def execute_sell(
        self,
        quotes: Dict[str, Dict],
        curr_date: str,
        curr_time: str,
        positions: List[XtPosition],
        held_days: Dict[str, int],
        max_prices: Dict[str, float],
        cache_history: Dict[str, pd.DataFrame],
        today_ticks: Dict[str, list] = None,
        extra_datas: Dict[str, any] = None,
    ) -> None:
        if today_ticks is None:
            today_ticks = {}

        if extra_datas is None:
            extra_datas = {}

        for position in positions:
            code = position.stock_code

            # 如果有数据且有持仓时间记录
            if (code in quotes) and (code in held_days):
                self.check_sell(
                    code=code,
                    quote=quotes[code],
                    curr_date=curr_date,
                    curr_time=curr_time,
                    position=position,
                    held_day=held_days[code],
                    max_price=max_prices[code] if code in max_prices else None,
                    history=cache_history[code] if code in cache_history else None,
                    ticks=today_ticks[code] if code in today_ticks else None,
                    extra=extra_datas[code] if code in extra_datas else None,
                )

    def check_sell(
        self, code: str, quote: Dict, curr_date: str, curr_time: str,
        position: XtPosition, held_day: int, max_price: Optional[float],
        history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        return False  # False 表示没有卖过，不阻挡其他Seller卖出
