from abc import ABC, abstractmethod


DEFAULT_STRATEGY_NAME = '空白策略'


class BaseDelegate(ABC):
    def __init__(self):
        self.callback = None

    @abstractmethod
    def check_asset(self):
        pass

    @abstractmethod
    def check_orders(self):
        pass

    @abstractmethod
    def check_positions(self):
        pass

    @abstractmethod
    def order_market_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ):
        pass

    @abstractmethod
    def order_market_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ):
        pass

    @abstractmethod
    def order_limit_open(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ):
        pass

    @abstractmethod
    def order_limit_close(
        self,
        code: str,
        price: float,
        volume: int,
        remark: str,
        strategy_name: str = DEFAULT_STRATEGY_NAME,
    ):
        pass

    @abstractmethod
    def order_cancel_all(self, strategy_name: str = DEFAULT_STRATEGY_NAME):
        pass

    @abstractmethod
    def order_cancel_buy(self, code: str, strategy_name: str = DEFAULT_STRATEGY_NAME):
        pass

    @abstractmethod
    def order_cancel_sell(self, code: str, strategy_name: str = DEFAULT_STRATEGY_NAME):
        pass
