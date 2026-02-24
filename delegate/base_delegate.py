import threading

from abc import ABC, abstractmethod
from tools.utils_cache import InfoItem, load_json, save_json


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

    @staticmethod
    def is_position_holding(position: dict) -> bool:
        return False

    @abstractmethod
    def get_holding_position_count(self, positions: list, only_stock: bool = False) -> int:
        return 0

    @abstractmethod
    def shutdown(self) -> None:
        pass


# -----------------------
# 持仓自动发现更新进缓存
# -----------------------
def update_position_held(lock: threading.Lock, delegate: BaseDelegate, path: str):
    with lock:
        positions = delegate.check_positions()
        if positions is None:
            print('[扫描仓位] 未能获取持仓数据')
        elif len(positions) <= 0:
            print('[扫描仓位] 当前空仓')
        else:
            position_codes = [position.stock_code for position in positions]
            print('[扫描仓位] 当前持仓', position_codes)

            held_info = load_json(path)

            # 添加未被缓存记录的持仓：默认当日买入
            for position in positions:
                if position.can_use_volume > 0:
                    if position.stock_code not in held_info.keys():
                        held_info[position.stock_code] = {InfoItem.DayCount: 0}

            # 删除已清仓的held_info记录
            holding_codes = list(held_info.keys())
            for code in holding_codes:
                if len(code) > 0 and code[0] != '_' and (code not in position_codes):
                    del held_info[code]

            save_json(path, held_info)
