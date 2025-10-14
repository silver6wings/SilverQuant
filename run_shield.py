import logging

from credentials import *

from tools.utils_basic import logging_init, is_symbol
from tools.utils_cache import *
from tools.utils_ding import DingMessager

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolWhiteIndexes as Pool
from trader.buyer import BaseBuyer as Buyer
from trader.seller_groups import ShieldGroupSeller as Seller


STRATEGY_NAME = '防御监控'
DING_MESSAGER = DingMessager(DING_SECRET, DING_TOKENS)
IS_PROD = False     # 生产环境标志：False 表示使用掘金模拟盘 True 表示使用QMT账户下单交易
IS_DEBUG = True     # 日志输出标记：控制台是否打印debug方法的输出

PATH_BASE = CACHE_PROD_PATH if IS_PROD else CACHE_TEST_PATH

PATH_ASSETS = PATH_BASE + '/assets.csv'         # 记录历史净值
PATH_DEAL = PATH_BASE + '/deal_hist.csv'        # 记录历史成交
PATH_HELD = PATH_BASE + '/positions.json'       # 记录持仓信息
PATH_MAXP = PATH_BASE + '/max_price.json'       # 记录建仓后历史最高
PATH_MINP = PATH_BASE + '/min_price.json'       # 记录建仓后历史最低
PATH_LOGS = PATH_BASE + '/logs.txt'             # 记录策略的历史日志
disk_lock = threading.Lock()                    # 操作磁盘文件缓存的锁
cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重


class PoolConf:
    white_indexes = []
    black_prompts = []

    # 忽略监控列表
    ignore_stocks = [
        '000001.SZ',  # 深交所股票尾部加.SZ
        '600000.SH',  # 上交所股票尾部加.SH
    ]


class SellConf:
    time_ranges = [['09:31', '11:30'], ['13:00', '14:57']]
    interval = 1                    # 扫描卖出间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.03            # 保证市价单成交的溢价，单位（元）

    hard_time_range = ['09:31', '14:57']
    earn_limit = 9.999              # 硬性止盈率
    risk_limit = 1 - 0.03           # 硬性止损率
    risk_tight = 0.003              # 硬性止损率每日上移

    # 利润从最高点回落卖出
    fall_time_range = ['09:31', '14:57']
    fall_from_top = [         # 至少保留收益范围
        (1.05, 9.99, 0.02),  # [2.900 % ~ 879.020 %)
        (1.02, 1.05, 0.05),  # [-3.100 % ~ -0.250 %)
    ]


# ======== 盘前 ========


def before_trade_day() -> None:
    # held_increase() -> None:
    update_position_held(disk_lock, my_delegate, PATH_HELD)
    if all_held_inc(disk_lock, PATH_HELD):
        logging.warning('===== 所有持仓计数 +1 =====')
        print(f'All held stock day +1!')

    # refresh_code_list() -> None:
    my_pool.refresh()
    positions = my_delegate.check_positions()
    hold_list = [position.stock_code for position in positions if is_symbol(position.stock_code)]
    full_list = my_pool.get_code_list() + hold_list
    target_list = [code for code in full_list if code not in PoolConf.ignore_stocks]

    my_suber.update_code_list(target_list)


# ======== 卖点 ========


def scan_sell(quotes: Dict, curr_date: str, curr_time: str, positions: List) -> None:
    max_prices, held_info = update_max_prices(disk_lock, quotes, positions, PATH_MAXP, PATH_MINP, PATH_HELD)
    my_seller.execute_sell(quotes, curr_date, curr_time, positions, held_info, max_prices, my_suber.cache_history)


# ======== 框架 ========


def execute_strategy(curr_date: str, curr_time: str, curr_seconds: str, curr_quotes: Dict) -> bool:
    positions = my_delegate.check_positions()

    for time_range in SellConf.time_ranges:
        if time_range[0] <= curr_time <= time_range[1]:
            if int(curr_seconds) % SellConf.interval == 0:
                scan_sell(curr_quotes, curr_date, curr_time, positions)

    return False


if __name__ == '__main__':
    logging_init(path=PATH_LOGS, level=logging.INFO)
    STRATEGY_NAME = STRATEGY_NAME if IS_PROD else STRATEGY_NAME + "[测]"
    print(f'正在启动 {STRATEGY_NAME}...')
    if IS_PROD:
        from delegate.xt_callback import XtCustomCallback
        from delegate.xt_delegate import XtDelegate

        my_callback = XtCustomCallback(
            account_id=QMT_ACCOUNT_ID,
            strategy_name=STRATEGY_NAME,
            ding_messager=DING_MESSAGER,
            disk_lock=disk_lock,
            path_deal=PATH_DEAL,
            path_held=PATH_HELD,
            path_max_prices=PATH_MAXP,
            path_min_prices=PATH_MINP,
        )
        my_delegate = XtDelegate(
            account_id=QMT_ACCOUNT_ID,
            client_path=QMT_CLIENT_PATH,
            callback=my_callback,
            ding_messager=DING_MESSAGER,
        )
    else:
        from delegate.gm_callback import GmCallback
        from delegate.gm_delegate import GmDelegate

        my_callback = GmCallback(
            account_id=QMT_ACCOUNT_ID,
            strategy_name=STRATEGY_NAME,
            ding_messager=DING_MESSAGER,
            disk_lock=disk_lock,
            path_deal=PATH_DEAL,
            path_held=PATH_HELD,
            path_max_prices=PATH_MAXP,
            path_min_prices=PATH_MINP,
        )
        my_delegate = GmDelegate(
            account_id=QMT_ACCOUNT_ID,
            callback=my_callback,
            ding_messager=DING_MESSAGER,
        )

    my_pool = Pool(
        account_id=QMT_ACCOUNT_ID,
        strategy_name=STRATEGY_NAME,
        parameters=PoolConf,
        ding_messager=DING_MESSAGER,
    )
    my_seller = Seller(
        strategy_name=STRATEGY_NAME,
        delegate=my_delegate,
        parameters=SellConf,
    )
    my_suber = XtSubscriber(
        account_id=QMT_ACCOUNT_ID,
        strategy_name=STRATEGY_NAME,
        delegate=my_delegate,
        path_deal=PATH_DEAL,
        path_assets=PATH_ASSETS,
        execute_strategy=execute_strategy,
        before_trade_day=before_trade_day,
        use_ap_scheduler=True,
        ding_messager=DING_MESSAGER,
        open_tick_memory_cache=True,
        open_today_deal_report=True,
        open_today_hold_report=True,
    )
    my_suber.start_scheduler()
