"""
https://sim.myquant.cn/sim/help/Python.html
"""
import datetime
from typing import List

from gmtrade.api import *
from gmtrade.pb.account_pb2 import Cash, Position, Order

from delegate.base_delegate import BaseDelegate
from delegate.gm_callback import GmCallback

from credentials import GM_ACCOUNT_ID, GM_CLIENT_TOKEN

from tools.utils_basic import code_to_gmsymbol, gmsymbol_to_code
from tools.utils_cache import StockNames
from tools.utils_ding import BaseMessager


DEFAULT_GM_SERVER_HOST = 'api.myquant.cn:9000'
DEFAULT_GM_STRATEGY_NAME = '模拟策略'


class GmAsset:
    def __init__(self, cash: Cash):
        self.account_type = 0
        self.account_id = cash.account_id
        self.cash = round(cash.available, 3)
        self.frozen_cash = round(cash.order_frozen, 3)
        self.market_value = round(cash.frozen, 3)
        self.total_asset = round(cash.nav, 3)


class GmOrder:
    def __init__(self, order: Order):
        self.account_id = order.account_id
        self.stock_code = gmsymbol_to_code(order.symbol)
        self.order_id = order.order_id
        self.order_volume = order.volume
        self.price = order.price
        self.order_type = order.order_type
        self.order_status = order.status


class GmPosition:
    def __init__(self, position: Position):
        self.account_id = position.account_id
        self.stock_code = gmsymbol_to_code(position.symbol)
        self.volume = position.volume
        self.can_use_volume = position.available
        self.open_price = position.vwap
        self.market_value = position.amount


class GmDelegate(BaseDelegate):
    def __init__(
        self,
        account_id: str = None,
        callback: GmCallback = None,
        ding_messager: BaseMessager = None,
    ):
        super().__init__()
        self.ding_messager = ding_messager
        self.stock_names = StockNames()

        self.account_id = '**' + str(account_id)[-4:]

        set_endpoint(DEFAULT_GM_SERVER_HOST)
        set_token(GM_CLIENT_TOKEN)

        self.account = account(account_id=GM_ACCOUNT_ID, account_alias='')
        login(self.account)

        if callback is not None:
            self.callback = callback
            self.callback.register_callback()

    def shutdown(self):
        self.callback.unregister_callback()

    def check_asset(self) -> GmAsset:
        cash: Cash = get_cash(self.account)
        return GmAsset(cash)

    def check_orders(self) -> List[GmOrder]:
        orders = get_orders(self.account)
        return [GmOrder(order) for order in orders]

    def check_positions(self) -> List[GmPosition]:
        positions = get_positions(self.account)
        return [GmPosition(position) for position in positions if position.volume > 0]

    def order_market_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_GM_STRATEGY_NAME,
    ):
        """
        [
            account_id: "189ca421-49db-11ef-9fa8-00163e022aa6"
            cl_ord_id: "83fe1b04-4afa-11ef-97f5-00163e022aa6"
            order_id: "83fe1b0b-4afa-11ef-97f5-00163e022aa6"
            ex_ord_id: "83fe1b0b-4afa-11ef-97f5-00163e022aa6"
            symbol: "SHSE.600000"
            side: 1
            position_effect: 1
            order_type: 2
            order_qualifier: 3
            status: 1
            order_style: 1
            volume: 100
            created_at {
              seconds: 1721962528
              nanos: 852249292
            }
            updated_at {
              seconds: 1721962528
              nanos: 852249292
            }
        ]
        """
        orders = order_volume(
            symbol=code_to_gmsymbol(code),
            price=price,
            volume=volume,
            side=OrderSide_Buy,
            order_type=OrderType_Market,
            order_qualifier=OrderQualifier_B5TC,
            position_effect=PositionEffect_Open,
        )
        print(f'[{remark}]{code}')
        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {remark}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 市买 {code}\n'
                f'{name} {volume}股 {price:.2f}元',
                '[MB]')
        return orders

    def order_market_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_GM_STRATEGY_NAME,
    ):
        orders = order_volume(
            symbol=code_to_gmsymbol(code),
            price=price,
            volume=volume,
            side=OrderSide_Sell,
            order_type=OrderType_Market,
            order_qualifier=OrderQualifier_B5TC,
            position_effect=PositionEffect_Close,
        )
        print(f'[{remark}]{code}')
        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {remark}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 市卖 {code}\n'
                f'{name} {volume}股 {price:.2f}元',
                '[MS]')
        return orders

    def order_limit_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_GM_STRATEGY_NAME,
    ):
        """
        [
            account_id: "189ca421-49db-11ef-9fa8-00163e022aa6"
            cl_ord_id: "83fe1b04-4afa-11ef-97f5-00163e022aa6"
            order_id: "83fe1b0b-4afa-11ef-97f5-00163e022aa6"
            ex_ord_id: "83fe1b0b-4afa-11ef-97f5-00163e022aa6"
            symbol: "SHSE.600000"
            side: 1
            position_effect: 1
            order_type: 2
            order_qualifier: 3
            status: 1
            order_style: 1
            volume: 100
            created_at {
              seconds: 1721962528
              nanos: 852249292
            }
            updated_at {
              seconds: 1721962528
              nanos: 852249292
            }
        ]
        """
        orders = order_volume(
            symbol=code_to_gmsymbol(code),
            price=price,
            volume=volume,
            side=OrderSide_Buy,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Open,
        )
        print(f'[{remark}]{code}')
        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {remark}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 限买 {code}\n'
                f'{name} {volume}股 {price:.2f}元',
                '[LB]')
        return orders

    def order_limit_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_GM_STRATEGY_NAME,
    ):
        orders = order_volume(
            symbol=code_to_gmsymbol(code),
            price=price,
            volume=volume,
            side=OrderSide_Sell,
            order_type=OrderType_Limit,
            position_effect=PositionEffect_Close,
        )
        print(f'[{remark}]{code}')
        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {remark}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 限卖 {code}\n'
                f'{name} {volume}股 {price:.2f}元',
                '[LS]')
        return orders

    def order_cancel_all(self, strategy_name: str = DEFAULT_GM_STRATEGY_NAME):
        order_cancel_all()

        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 全撤\n'
                '[CA]')

    # order_1 = {'symbol': 'SHSE.600000', 'cl_ord_id': 'cl_ord_id_1', 'price': 11, 'side': 1, 'order_type': 1}
    # order_2 = {'symbol': 'SHSE.600004', 'cl_ord_id': 'cl_ord_id_2', 'price': 11, 'side': 1, 'order_type': 1}
    # orders = [order_1, order_2]
    # order_cancel(wait_cancel_orders=orders)

    def order_cancel_buy(self, code: str, strategy_name: str = DEFAULT_GM_STRATEGY_NAME):
        orders = get_orders()
        candidate = []
        for order in orders:
            if order.side == OrderSide_Buy:
                candidate.append({
                    'symbol': code_to_gmsymbol(code),
                    'cl_ord_id': order.cl_ord_id,
                })
        order_cancel(candidate)

        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {name}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 撤买 {code}\n'
                '[CB]')

    def order_cancel_sell(self, code: str, strategy_name: str = DEFAULT_GM_STRATEGY_NAME):
        orders = get_orders()
        candidate = []
        for order in orders:
            if order.side == OrderSide_Sell:
                candidate.append({
                    'symbol': code_to_gmsymbol(code),
                    'cl_ord_id': order.cl_ord_id,
                })
        order_cancel(candidate)

        if self.ding_messager is not None:
            name = self.stock_names.get_name(code)
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{strategy_name} {name}\n'
                f'{datetime.datetime.now().strftime("%H:%M:%S")} 撤卖 {code}\n'
                '[CS]')

    @staticmethod
    def is_position_holding(position: GmPosition) -> bool:
        return position.volume > 0


    def get_holding_position_count(self, positions: List[GmPosition]) -> int:
        return sum(1 for position in positions if self.is_position_holding(position))
