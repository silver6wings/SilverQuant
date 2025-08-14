from trader.seller_components import *


class GroupSellers:
    def __init__(self):
        pass

    def group_init(self, strategy_name, delegate, parameters):
        for parent in self.__class__.__bases__:
            if parent.__name__ != 'GroupSellers':
                parent.__init__(self, strategy_name, delegate, parameters)
        print('>> 初始化完成')

    def group_check_sell(
            self, code: str, quote: Dict, curr_date: str, curr_time: str,
            position: XtPosition, held_day: int, max_price: Optional[float],
            history: Optional[pd.DataFrame], ticks: Optional[list[list]], extra: any,
    ) -> bool:
        sold = False
        for parent in self.__class__.__bases__:
            if parent.__name__ != 'GroupSellers':
                if sold:
                    break
                else:
                    sold = parent.check_sell(
                        self, code=code, quote=quote, curr_date=curr_date, curr_time=curr_time,
                        position=position, held_day=held_day, max_price=max_price,
                        history=history, ticks=ticks, extra=extra,
                    )
        return sold


# 传统卖出
class ClassicGroupSeller(GroupSellers, HardSeller, FallSeller, ReturnSeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 均线卖出
class ClassicMAGroupSeller(GroupSellers, HardSeller, FallSeller, ReturnSeller, MASeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 监控卖出
class ShieldGroupSeller(GroupSellers, HardSeller, FallSeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 问财卖出
class WencaiGroupSeller(GroupSellers, HardSeller, FallSeller, ReturnSeller, SwitchSeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# Deepseek
class DeepseekGroupSeller(GroupSellers, HardSeller, SwitchSeller, FallSeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 龙抬头
class LTT2GroupSeller(GroupSellers, HardSeller, OpenDaySeller, SwitchSeller, ReturnSeller, CCISeller, MASeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 三倍量突破
class T3BLGroupSeller(GroupSellers, HardSeller, SwitchSeller, FallSeller, MASeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 平台绿缩
class PTLSGroupSeller(GroupSellers, HardSeller, UppingBlocker, FallSeller, ReturnSeller, SwitchSeller, MASeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)


# 金雀突破
class JQTPGroupSeller(GroupSellers, FallSeller, DropSeller, HardSeller, WRSeller, ReturnSeller):
    def __init__(self, strategy_name, delegate, parameters):
        super().__init__()
        self.group_init(strategy_name, delegate, parameters)

    def check_sell(self, code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra):
        self.group_check_sell(code, quote, curr_date, curr_time, position, held_day, max_price, history, ticks, extra)
