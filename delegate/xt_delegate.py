import time
from threading import Thread
from typing import List, Optional

from xtquant import xtconstant
from xtquant.xtconstant import STOCK_BUY, STOCK_SELL
from xtquant.xtdata import get_client, get_full_tick
from xtquant.xttrader import XtQuantTrader
from xtquant.xttype import StockAccount, XtPosition, XtOrder, XtAsset

from credentials import *
from tools.utils_basic import get_code_exchange
from delegate.base_delegate import BaseDelegate
from delegate.xt_callback import XtDefaultCallback


default_client_path = QMT_CLIENT_PATH
default_account_id = QMT_ACCOUNT_ID

default_reconnect_duration = 60
default_wait_duration = 15


class XtDelegate(BaseDelegate):
    def __init__(
        self,
        account_id: str = None,
        client_path: str = None,
        callback: object = None,
        keep_run: bool = True,
    ):
        super().__init__()
        self.xt_trader: Optional[XtQuantTrader] = None

        if client_path is None:
            client_path = default_client_path
        self.path = client_path

        if account_id is None:
            account_id = default_account_id
        self.account = StockAccount(account_id=account_id, account_type='STOCK')
        self.callback = callback
        self.connect(self.callback)
        if keep_run:
            # 保证QMT持续连接
            Thread(target=self.keep_connected).start()

    def connect(self, callback: object) -> (XtQuantTrader, bool):
        session_id = int(time.time())  # 生成session id 整数类型 同时运行的策略不能重复
        print("生成临时 session_id: ", session_id)
        self.xt_trader = XtQuantTrader(self.path, session_id)

        if callback is None:
            callback = XtDefaultCallback()

        callback.delegate = self
        self.xt_trader.register_callback(callback)

        self.xt_trader.start()  # 启动交易线程

        # 建立交易连接，返回0表示连接成功
        print('正在建立交易连接...', end='')
        connect_result = self.xt_trader.connect()
        print(f'返回值：{connect_result}...', end='')
        if connect_result != 0:
            print('失败!')
            self.xt_trader = None
            return None, False
        print('成功!')

        # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
        print('正在订阅主推回调...', end='')
        subscribe_result = self.xt_trader.subscribe(self.account)
        print(f'返回值：{subscribe_result}...', end='')
        if subscribe_result != 0:
            print('失败!')
            self.xt_trader = None
            return None, False
        print('成功!')

        print('连接完毕。')
        return self.xt_trader, True

    def reconnect(self) -> None:
        if self.xt_trader is None:
            print('开始重连交易接口')
            _, success = self.connect(self.callback)
            if success:
                print('交易接口重连成功')
        # else:
        #     print('无需重连交易接口')
        #     pass

    def keep_connected(self) -> None:
        while True:
            time.sleep(default_reconnect_duration)
            self.reconnect()

    def shutdown(self):
        self.xt_trader.stop()
        self.xt_trader = None

    def order_submit(
        self,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str,
        order_remark: str,
    ) -> bool:
        if self.xt_trader is not None:
            self.xt_trader.order_stock(
                account=self.account,
                stock_code=stock_code,
                order_type=order_type,
                order_volume=order_volume,
                price_type=price_type,
                price=price,
                strategy_name=strategy_name,
                order_remark=order_remark,
            )
            return True
        else:
            return False

    def order_submit_async(
        self,
        stock_code: str,
        order_type: int,
        order_volume: int,
        price_type: int,
        price: float,
        strategy_name: str,
        order_remark: str,
    ) -> bool:
        if self.xt_trader is not None:
            self.xt_trader.order_stock_async(
                account=self.account,
                stock_code=stock_code,
                order_type=order_type,
                order_volume=order_volume,
                price_type=price_type,
                price=price,
                strategy_name=strategy_name,
                order_remark=order_remark,
            )
            return True
        else:
            return False

    def order_cancel(self, order_id) -> int:
        cancel_result = self.xt_trader.cancel_order_stock(self.account, order_id)
        return cancel_result

    def order_cancel_async(self, order_id) -> int:
        cancel_result = self.xt_trader.cancel_order_stock_async(self.account, order_id)
        return cancel_result

    def check_asset(self) -> XtAsset:
        if self.xt_trader is not None:
            return self.xt_trader.query_stock_asset(self.account)
        else:
            raise Exception('xt_trader为空')

    def check_order(self, order_id) -> XtOrder:
        if self.xt_trader is not None:
            return self.xt_trader.query_stock_order(self.account, order_id)
        else:
            raise Exception('xt_trader为空')

    def check_orders(self, cancelable_only: bool = False) -> List[XtOrder]:
        if self.xt_trader is not None:
            orders = self.xt_trader.query_stock_orders(self.account, cancelable_only)
            if cancelable_only:
                '''
                参考委托XtOrder数据结构说明，price_type为柜台返回的枚举值，且xtquant库未定义枚举值常量名。
                以下常见情况不可取消委托，而GJ等柜台返回数据包含这部分数据需通过order的price_type值过滤掉。
                BROKER_PRICE_PROP_SUBSCRIBE			54	申购
                BROKER_PRICE_PROP_FUND_ENTRUST		79	基金申赎
                BROKER_PRICE_PROP_ETF				81	ETF申购
                BROKER_PRICE_PROP_DEBT_CONVERSION	91	债转股
                '''
                return [order for order in orders if order.price_type not in [54, 79, 81, 91]]
            else:
                return orders
        else:
            raise Exception('xt_trader为空')

    def check_positions(self) -> List[XtPosition]:
        if self.xt_trader is not None:
            return self.xt_trader.query_stock_positions(self.account)
        else:
            raise Exception('xt_trader为空')

    def order_market_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = 'non-name',
    ):
        price_type = xtconstant.LATEST_PRICE

        if get_code_exchange(code) == 'SZ':
            price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL
            price = -1
        if get_code_exchange(code) == 'SH':
            price_type = xtconstant.MARKET_PEER_PRICE_FIRST
            price = price

        self.order_submit(
            stock_code=code,
            order_type=xtconstant.STOCK_BUY,
            order_volume=volume,
            price_type=price_type,
            price=price,
            strategy_name=strategy_name,
            order_remark=remark,
        )

    def order_market_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = 'non-name',
    ):
        price_type = xtconstant.LATEST_PRICE

        if get_code_exchange(code) == 'SZ':
            price_type = xtconstant.MARKET_SZ_CONVERT_5_CANCEL
            price = -1
        if get_code_exchange(code) == 'SH':
            price_type = xtconstant.MARKET_PEER_PRICE_FIRST
            price = price

        self.order_submit(
            stock_code=code,
            order_type=xtconstant.STOCK_SELL,
            order_volume=volume,
            price_type=price_type,
            price=price,
            strategy_name=strategy_name,
            order_remark=remark,
        )

    def order_limit_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = 'non-name',
    ):
        self.order_submit(
            stock_code=code,
            price=price,
            order_volume=volume,
            order_type=xtconstant.STOCK_BUY,
            price_type=xtconstant.FIX_PRICE,
            strategy_name=strategy_name,
            order_remark=remark,
        )

    def order_limit_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = 'non-name',
    ):
        self.order_submit(
            stock_code=code,
            price=price,
            order_volume=volume,
            order_type=xtconstant.STOCK_SELL,
            price_type=xtconstant.FIX_PRICE,
            strategy_name=strategy_name,
            order_remark=remark,
        )

    # # 已报
    # ORDER_REPORTED = 50
    # # 已报待撤
    # ORDER_REPORTED_CANCEL = 51
    # # 部成待撤
    # ORDER_PARTSUCC_CANCEL = 52
    # # 部撤
    # ORDER_PART_CANCEL = 53

    def order_cancel_all(self):
        orders = self.check_orders(cancelable_only=True)
        for order in orders:
            self.order_cancel_async(order.order_id)

    def order_cancel_buy(self, code: str):
        orders = self.check_orders(cancelable_only=True)
        for order in orders:
            if order.stock_code == code and order.order_type == STOCK_BUY:
                self.order_cancel_async(order.order_id)

    def order_cancel_sell(self, code: str):
        orders = self.check_orders(cancelable_only=True)
        for order in orders:
            if order.stock_code == code and order.order_type == STOCK_SELL:
                self.order_cancel_async(order.order_id)

    def check_ipo_data(self) -> dict:
        if self.xt_trader is not None:
            return self.xt_trader.query_ipo_data()
        else:
            raise Exception('xt_trader为空')
        
    def check_new_purchase_limit(self) -> dict:
        if self.xt_trader is not None:
            return self.xt_trader.query_new_purchase_limit(self.account)
        else:
            raise Exception('xt_trader为空')


def is_position_holding(position: XtPosition) -> bool:
    return position.volume > 0


def get_holding_position_count(positions: List[XtPosition]) -> int:
    return sum(1 for position in positions if is_position_holding(position))


def xt_stop_exit():
    import time
    client = get_client()
    while True:
        time.sleep(default_wait_duration)
        if not client.is_connected():
            print('行情服务连接断开...')


def xt_get_ticks(code_list: list[str]):
    # http://docs.thinktrader.net/pages/36f5df/#%E8%8E%B7%E5%8F%96%E5%85%A8%E6%8E%A8%E6%95%B0%E6%8D%AE
    return get_full_tick(code_list)


if __name__ == '__main__':
    # my_delegate = XtDelegate()
    # my_delegate.xt_trader.run_forever()
    # my_delegate.xt_trader.stop()
    print(xt_get_ticks(['000001.SZ']))
