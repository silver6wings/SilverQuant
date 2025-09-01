import logging

from credentials import *

from tools.utils_basic import logging_init, is_symbol
from tools.utils_cache import *
from tools.utils_ding import DingMessager
from tools.utils_remote import DataSource

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolBlackEmpty as Pool
# from trader.buyer import BaseBuyer as Buyer
from trader.seller_my_groups import LHSGroupSeller as Seller


STRATEGY_NAME = '均线防御'
DING_MESSAGER = DingMessager(DING_SECRET, DING_TOKENS)

HISTORY_DATA_SOURCE = DataSource.AKSHARE
IS_PROD = True
IS_DEBUG = True

PATH_BASE = CACHE_PROD_PATH if IS_PROD else CACHE_TEST_PATH

PATH_ASSETS = PATH_BASE + '/assets.csv'         # 记录历史净值
PATH_DEAL = PATH_BASE + '/deal_hist.csv'        # 记录历史成交
PATH_HELD = PATH_BASE + '/held_days.json'       # 记录持仓日期
PATH_MAXP = PATH_BASE + '/max_price.json'       # 记录建仓后历史最高
PATH_MINP = PATH_BASE + '/min_price.json'       # 记录建仓后历史最低
PATH_LOGS = PATH_BASE + '/logs.txt'             # 用来存储选股和委托操作
PATH_INFO = PATH_BASE + '/tmp_{}.pkl'           # 用来缓存当天的指标信息

disk_lock = threading.Lock()           # 操作磁盘文件缓存的锁

cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重


def debug(*args, **kwargs):
    if IS_DEBUG:
        print(*args, **kwargs)


class PoolConf:
    # white_prefixes = {'00', '60'}
    # white_none_st = True
    # black_prompts = []

    # 忽略监控列表
    ignore_stocks = [
        # '000001.SZ',        # 深交所股票尾部加.SZ
        # '600000.SH',        # 上交所股票尾部加.SH
    ]

    day_count = 300         # 200个足够算出周期为120的均线数据
    price_adjust = 'qfq'    # 历史价格复权
    columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']


# class BuyConf:
#     time_ranges = [
#         ['09:30:05', '10:10:00']
#     ]
#     interval = 3
#     order_premium = 0.05    # 保证成功买入成交的溢价
#
#     slot_count = 10         # 持股数量
#     slot_capacity = 10000   # 每个仓的资金上限
#     once_buy_limit = 10     # 单次选股最多买入股票数量（若单次未买进当日不会再买这只


class SellConf:
    time_ranges = [
        ['09:45:00', '11:30:00'],
        ['13:00:00', '14:57:00'],
    ]
    interval = 3                    # 扫描卖出间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.05            # 保证市价单成交的溢价，单位（元）

    hard_time_range = ['09:45', '14:57']
    earn_limit = 9.999              # 硬性止盈率
    risk_limit = 1 - 0.07           # 硬性止损率
    risk_tight = 0.005              # 硬性止损率每日上移

    ma_time_range = ['10:30', '14:57']
    ma_above = 10


# ======== 盘前 ========


def before_trade_day() -> None:
    print('held_increase()')
    update_position_held(disk_lock, my_delegate, PATH_HELD)
    if all_held_inc(disk_lock, PATH_HELD):
        logging.warning('===== 所有持仓计数 +1 =====')
        print(f'All held stock day +1!')

    print('refresh_code_list()')
    my_pool.refresh()
    positions = my_delegate.check_positions()
    hold_list = [position.stock_code for position in positions if is_symbol(position.stock_code)]
    # full_list = my_pool.get_code_list() + hold_list
    # target_list = [code for code in full_list if code not in PoolConf.ignore_stocks]
    my_suber.update_code_list(hold_list)

    print('prepare_history()')
    # 获取历史起止日期
    now = datetime.datetime.now()
    start = get_prev_trading_date(now, PoolConf.day_count)
    end = get_prev_trading_date(now, 1)

    my_suber.download_cache_history(
        cache_path='',
        code_list=hold_list,
        start=start,
        end=end,
        adjust=PoolConf.price_adjust,
        columns=PoolConf.columns,
        data_source=HISTORY_DATA_SOURCE,
    )


# ======== 卖点 ========


def scan_sell(quotes: Dict, curr_date: str, curr_time: str, positions: List) -> None:
    max_prices, held_days = update_max_prices(disk_lock, quotes, positions, PATH_MAXP, PATH_MINP, PATH_HELD, False)
    my_seller.execute_sell(quotes, curr_date, curr_time, positions, held_days, max_prices, my_suber.cache_history)


# ======== 框架 ========


def execute_strategy(curr_date: str, curr_time: str, curr_seconds: str, curr_quotes: Dict) -> bool:
    positions = my_delegate.check_positions()
    can_clear = False

    curr_ts = f"{curr_time}:{curr_seconds}"

    for time_range in SellConf.time_ranges:
        if time_range[0] <= curr_ts <= time_range[1]:
            if int(curr_seconds) % SellConf.interval == 0:
                scan_sell(curr_quotes, curr_date, curr_time, positions)

    return can_clear


if __name__ == '__main__':
    logging_init(path=PATH_LOGS, level=logging.INFO)
    STRATEGY_NAME = STRATEGY_NAME if IS_PROD else STRATEGY_NAME + "(测)"
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
    # my_buyer = Buyer(
    #     account_id=QMT_ACCOUNT_ID,
    #     strategy_name=STRATEGY_NAME,
    #     delegate=my_delegate,
    #     parameters=BuyConf,
    # )
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
