import os
import sys

# 添加父目录到模块搜索路径，方便设置开机启动任务
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from credentials import *
from tools.utils_basic import logging_init, is_symbol, debug
from tools.utils_cache import *
from tools.utils_ding import DingMessager
from tools.utils_remote import get_wencai_codes, get_mootdx_quotes

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolBlackWencai as Pool
from trader.buyer import BaseBuyer as Buyer
from trader.seller_groups import WencaiGroupSeller as Seller

from selector.select_wencai import get_prompt


STRATEGY_NAME = '问财TDX'
SELECT_PROMPT = get_prompt()
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
cache_history: Dict[str, pd.DataFrame] = {}     # 记录历史日线行情的信息 { code: DataFrame }


class PoolConf:
    black_prompts = ['ST', '退市']


class BuyConf:
    time_ranges = [['14:50', '14:57']]

    # wencai 尽可能时间长些，不然会被封IP
    interval = 30           # 扫描买入间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.02    # 保证市价单成交的溢价，单位（元）

    slot_count = 50         # 持股数量上限
    slot_capacity = 10000   # 每个仓的资金上限
    daily_buy_max = 10      # 单日买入股票上限
    once_buy_limit = 10     # 单次选股最多买入股票数量
    inc_limit = 1.20        # 相对于昨日收盘的涨幅限制


class SellConf:
    time_ranges = [['09:31', '11:30'], ['13:00', '14:57']]
    interval = 15                   # 扫描卖出间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.02            # 保证市价单成交的溢价，单位（元）

    hard_time_range = ['09:31', '14:57']
    earn_limit = 9.999              # 硬性止盈率
    risk_limit = 1 - 0.03           # 硬性止损率
    risk_tight = 0.002              # 硬性止损率每日上移

    switch_time_range = ['14:30', '14:57']
    switch_hold_days = 5            # 持仓天数
    switch_demand_daily_up = 0.002  # 换仓上限乘数

    # 利润从最高点回落卖出
    fall_time_range = ['09:31', '14:57']
    fall_from_top = [
        (1.08, 9.99, 0.02),
        (1.02, 1.08, 0.05),
    ]

    # 涨幅超过建仓价xA，并小于建仓价xB 时，回撤涨幅的C倍卖出
    # (A, B, C)
    return_time_range = ['09:31', '14:57']
    return_of_profit = [
        (1.20, 9.99, 0.100),
        (1.08, 1.20, 0.200),
        (1.05, 1.08, 0.300),
        (1.03, 1.05, 0.500),
        (1.02, 1.03, 0.800),
    ]


# ======== 盘前 ========


def before_trade_day() -> None:
    # held_increase() -> None:
    if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
        return

    update_position_held(disk_lock, my_delegate, PATH_HELD)
    if all_held_inc(disk_lock, PATH_HELD):
        logging.warning('===== 所有持仓计数 +1 =====')
        print(f'All held stock day +1!')

    # refresh_code_list() -> None:
    my_pool.refresh()
    positions = my_delegate.check_positions()
    hold_list = [position.stock_code for position in positions if is_symbol(position.stock_code)]
    my_suber.update_code_list(hold_list)


# ======== 买点 ========


def pull_stock_codes() -> List[str]:
    codes_wencai = get_wencai_codes([SELECT_PROMPT])
    codes_top = []

    for code in codes_wencai:
        if code not in my_pool.cache_blacklist:
            codes_top.append(code)

    return codes_top


def check_stock_codes(selected_codes: list[str], quotes: Dict) -> dict[str, dict]:
    selections = {}

    for code in selected_codes:
        if code not in quotes:
            debug(code, f'本次quotes没数据')
            continue

        quote = quotes[code]
        curr_price = quote['lastPrice']

        last_close = quote['lastClose']
        if curr_price > last_close * BuyConf.inc_limit:
            continue

        selections[code] = {
            'price': max(quote['askPrice'] + [quote['lastPrice']]),
            'lastClose': quote['lastClose'],
        }

    return selections


def scan_buy(quotes: Dict, curr_date: str, positions: List) -> None:
    selected_codes = pull_stock_codes()
    print(selected_codes, quotes)

    selections = {}
    if selected_codes is not None and len(selected_codes) > 0:
        once_quotes = get_mootdx_quotes(selected_codes)
        selections = check_stock_codes(selected_codes, once_quotes)

    global cache_selected
    cache_selected = my_buyer.buy_selections(selections, cache_selected, curr_date, positions)


# ======== 卖点 ========


def scan_sell(quotes: Dict, curr_date: str, curr_time: str, positions: List) -> None:
    hold_list = [position.stock_code for position in positions if is_symbol(position.stock_code)]
    tdx_quotes = get_mootdx_quotes(hold_list)
    print(f'[{hold_list}|{len(tdx_quotes.keys())}|{len(quotes)}]', end='')

    max_prices, held_info = update_max_prices(disk_lock, tdx_quotes, positions, PATH_MAXP, PATH_MINP, PATH_HELD)
    my_seller.execute_sell(tdx_quotes, curr_date, curr_time, positions, held_info, max_prices, my_suber.cache_history)


# ======== 框架 ========


def execute_strategy(curr_date: str, curr_time: str, curr_seconds: str, curr_quotes: Dict) -> bool:
    positions = my_delegate.check_positions()

    for time_range in SellConf.time_ranges:
        if time_range[0] <= curr_time <= time_range[1]:
            if int(curr_seconds) % SellConf.interval == 0:
                scan_sell(curr_quotes, curr_date, curr_time, positions)

    for time_range in BuyConf.time_ranges:
        if time_range[0] <= curr_time <= time_range[1]:
            if int(curr_seconds) % BuyConf.interval == 0:
                scan_buy(curr_quotes, curr_date, positions)
                return True

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
    my_buyer = Buyer(
        strategy_name=STRATEGY_NAME,
        delegate=my_delegate,
        parameters=BuyConf,
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
        use_outside_data=True,
        use_ap_scheduler=True,
        ding_messager=DING_MESSAGER,
        open_today_deal_report=True,
        open_today_hold_report=True,
    )
    my_suber.start_scheduler()
