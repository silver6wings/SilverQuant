import threading
import datetime
import logging

from typing import Callable

from xtquant import xtconstant
from xtquant.xttrader import XtQuantTraderCallback
from xtquant.xttype import XtOrder, XtTrade, XtOrderError, XtCancelError, XtOrderResponse, XtCancelOrderResponse

from tools.utils_cache import record_deal, new_held, del_key, del_held_day, StockNames
from tools.utils_ding import BaseMessager


class XtBaseCallback(XtQuantTraderCallback):
    def __init__(self):
        self.delegate = None

    def on_disconnected(self):
        # deprecated
        print(datetime.datetime.now(), '连接丢失，断线重连中...')
        if self.delegate is not None:
            self.delegate.xt_trader = None


class XtDefaultCallback(XtBaseCallback):
    def on_stock_trade(self, trade: XtTrade):
        print(
            datetime.datetime.now(),
            f'成交回调 id:{trade.order_id} code:{trade.stock_code} remark:{trade.order_remark}',
        )

    def on_stock_order(self, order: XtOrder):
        print(
            datetime.datetime.now(),
            f'委托回调 id:{order.order_id} code:{order.stock_code} remark:{order.order_remark}',
        )

    def on_order_stock_async_response(self, res: XtOrderResponse):
        print(
            datetime.datetime.now(),
            f'异步委托回调 id:{res.order_id} sysid:{res.error_msg} remark:{res.order_remark}',
        )

    def on_order_error(self, order_error: XtOrderError):
        print(
            datetime.datetime.now(),
            f'委托报错回调 id:{order_error.order_id} error_id:{order_error.error_id} error_msg:{order_error.error_msg}',
        )

    def on_cancel_order_stock_async_response(self, res: XtCancelOrderResponse):
        print(
            datetime.datetime.now(),
            f'异步撤单回调 id:{res.order_id} sysid:{res.order_sysid} result:{res.cancel_result}',
        )

    def on_cancel_error(self, cancel_error: XtCancelError):
        print(
            datetime.datetime.now(),
            f'撤单报错回调 id:{cancel_error.order_id} error_id:{cancel_error.error_id} error_msg:{cancel_error.error_msg}',
        )

    # def on_account_status(self, status: XtAccountStatus):
    #     print(
    #         datetime.datetime.now(),
    #         f'账号查询回调: id:{status.account_id} type:{status.account_type} status:{status.status} ',
    #     )


class XtCustomCallback(XtBaseCallback):
    def __init__(
        self,
        account_id: str,
        strategy_name: str,
        ding_messager: BaseMessager,
        disk_lock: threading.Lock,
        path_deal: str,
        path_held: str,
        path_max_prices: str,
        path_min_prices: str,
        stock_traded_callback: Callable = None,
    ):
        super().__init__()
        self.account_id = '**' + str(account_id)[-4:]
        self.strategy_name = strategy_name
        self.ding_messager = ding_messager
        self.disk_lock = disk_lock
        self.path_deal = path_deal
        self.path_held = path_held
        self.path_max_prices = path_max_prices
        self.path_min_prices = path_min_prices

        self.stock_traded_callback = stock_traded_callback

        self.stock_names = StockNames()

    def record_order(self, order_time: str, code: str, price: float, volume: int, side: str, remark: str):
        record_deal(
            lock=self.disk_lock,
            path=self.path_deal,
            timestamp=order_time,
            code=code,
            name=self.stock_names.get_name(code),
            order_type=side,
            remark=remark,
            price=round(price, 3),
            volume=volume,
        )

    def on_stock_trade(self, trade: XtTrade):
        stock_code = trade.stock_code
        traded_volume = trade.traded_volume
        traded_price = trade.traded_price
        order_remark = trade.order_remark
        name = self.stock_names.get_name(stock_code)

        if trade.order_type == xtconstant.STOCK_SELL:
            del_held_day(self.disk_lock, self.path_held, stock_code)
            del_key(self.disk_lock, self.path_max_prices, stock_code)
            del_key(self.disk_lock, self.path_min_prices, stock_code)

            # traded_time = trade.traded_time
            # self.record_order(
            #     order_datetime=traded_time,
            #     code=stock_code,
            #     price=traded_price,
            #     volume=traded_volume,
            #     side='卖出成交',
            #     remark=order_remark,
            # )

            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name} {order_remark}\n'
                    f'{datetime.datetime.now().strftime("%H:%M:%S")} 卖成 {stock_code}\n'
                    f'{name} {traded_volume}股 {traded_price:.2f}元',
                    '[SOLD]')

        elif trade.order_type == xtconstant.STOCK_BUY:
            new_held(self.disk_lock, self.path_held, [stock_code])

            # traded_time = trade.traded_time
            # self.record_order(
            #     order_datetime=traded_time,
            #     code=stock_code,
            #     price=traded_price,
            #     volume=traded_volume,
            #     side='买入成交',
            #     remark=order_remark,
            # )

            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name} {order_remark}\n'
                    f'{datetime.datetime.now().strftime("%H:%M:%S")} 买成 {stock_code}\n'
                    f'{name} {traded_volume}股 {traded_price:.2f}元',
                    '[BOUGHT]')

        if self.stock_traded_callback is not None:
            self.stock_traded_callback(trade)

        return

    def on_order_stock_async_response(self, res: XtOrderResponse):
        log = f'异步下单委托 {res.order_id} msg:{res.error_msg} remark:{res.order_remark}',
        logging.warning(log)

    def on_order_error(self, err: XtOrderError):
        log = f'异步下单出错 {err.order_id} msg:{err.error_msg} remark:{err.order_remark} '
        logging.warning(log)

    def on_cancel_order_stock_async_response(self, res: XtCancelOrderResponse):
        log = f'异步撤单委托 {res.order_id} msg:{res.error_msg} result:{res.cancel_result} ',
        logging.warning(log)
        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name} 撤单成功\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} {res.cancel_result}\n'
                '[CANCEL]')

    def on_cancel_error(self, cancel_error: XtCancelError):
        log = f'异步撤单出错 {cancel_error.order_id} msg:{cancel_error.error_msg}',
        logging.warning(log)
