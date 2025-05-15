import datetime
import threading
from typing import Optional

from gmtrade.api import *
from gmtrade.pb.account_pb2 import Order, ExecRpt, AccountStatus

from tools.utils_basic import gmsymbol_to_code
from tools.utils_cache import record_deal, new_held, del_key, StockNames
from tools.utils_ding import DingMessager


class GmCallback:
    def __init__(
        self,
        account_id: str,
        strategy_name: str,
        ding_messager: DingMessager,
        disk_lock: threading.Lock,
        path_deal: str,
        path_held: str,
        path_max_prices: str,
        path_min_prices: str,
        debug: bool = False,
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

        self.stock_names = StockNames()
        self.debug: bool = debug

        GmCache.gm_callback = self

    @staticmethod
    def register_callback():
        file_name = str('delegate.gm_callback.py').split('\\')[-1]
        # print(file_name)
        status = start(filename=file_name)

        if status == 0:
            print(f'[掘金]:订阅回调成功')
        else:
            print(f'[掘金]:订阅回调失败')
            # stop()

    @staticmethod
    def unregister_callback():
        print(f'[掘金]:取消订阅回调')
        # stop()

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

    def on_execution_report(self, rpt: ExecRpt):
        """
            account_id: "189ca421-49db-11ef-9fa8-00163e022aa6"
            account_name: "189ca421-49db-11ef-9fa8-00163e022aa6"
            cl_ord_id: "83fe1b04-4afa-11ef-97f5-00163e022aa6"
            order_id: "83fe1b0b-4afa-11ef-97f5-00163e022aa6"
            exec_id: "84287d81-4afa-11ef-97f5-00163e022aa6"
            symbol: "SHSE.600000"
            position_effect: 1
            side: 1
            exec_type: 15
            price: 8.300000190734863
            volume: 100
            amount: 830.0000190734863
            created_at {
              seconds: 1721962529
              nanos: 130021792
            }
            cost: 830.0000190734863
        """
        pass

        # stock_code = gmsymbol_to_code(rpt.symbol)
        # traded_volume = rpt.volume
        # traded_price = float(rpt.price)
        # traded_time = rpt.created_at.seconds

        # if rpt.side == OrderSide_Buy:
        #     self.record_order(
        #         order_time=traded_time,
        #         code=stock_code,
        #         price=traded_price,
        #         volume=traded_volume,
        #         side='买入成交',
        #         remark='',
        #     )

        # if rpt.side == OrderSide_Sell:
        #     self.record_order(
        #         order_time=traded_time,
        #         code=stock_code,
        #         price=traded_price,
        #         volume=traded_volume,
        #         side='卖出成交',
        #         remark='',
        #     )

    def on_order_status(self, order: Order):
        if order.status == OrderStatus_Rejected:
            self.ding_messager.send_text_as_md(f'订单已拒绝:{order.symbol} {order.ord_rej_reason_detail}')

        elif order.status == OrderStatus_Filled:
            stock_code = gmsymbol_to_code(order.symbol)
            traded_volume = order.volume
            traded_price = order.price

            if order.side == OrderSide_Sell:
                del_key(self.disk_lock, self.path_held, stock_code)
                del_key(self.disk_lock, self.path_max_prices, stock_code)
                del_key(self.disk_lock, self.path_min_prices, stock_code)

                name = self.stock_names.get_name(stock_code)
                self.ding_messager.send_text_as_md(
                    f'{datetime.datetime.now().strftime("%H:%M:%S")} 卖出成交 {stock_code}\n'
                    f'{name} {traded_volume}股 {traded_price:.2f}元',
                    '[SELL]')

            if order.side == OrderSide_Buy:
                new_held(self.disk_lock, self.path_held, [stock_code])

                name = self.stock_names.get_name(stock_code)
                self.ding_messager.send_text_as_md(
                    f'{datetime.datetime.now().strftime("%H:%M:%S")} 买入成交 {stock_code}\n'
                    f'{name} {traded_volume}股 {traded_price:.2f}元',
                    '[BUY]')

        else:
            print(order.status, order.symbol)


class GmCache:
    gm_callback: Optional[GmCallback] = None


def on_trade_data_connected():
    print('[掘金回调]:交易服务已连接')


def on_trade_data_disconnected():
    print('\n[掘金回调]:交易服务已断开')


def on_account_status(account_status: AccountStatus):
    print('[掘金回调]:账户状态已变化')
    print(f'on_account_status status={account_status}')


def on_execution_report(rpt: ExecRpt):
    # print('[掘金回调]:成交状态已变化')
    GmCache.gm_callback.on_execution_report(rpt)


def on_order_status(order: Order):
    # print('[掘金回调]:订单状态已变')
    GmCache.gm_callback.on_order_status(order)
