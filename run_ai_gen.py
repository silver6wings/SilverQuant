import logging

from credentials import *

from tools.utils_basic import logging_init, is_symbol, debug
from tools.utils_cache import *
from tools.utils_ding import DingMessager
from tools.utils_remote import DataSource, ExitRight, concat_ak_quote_dict

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolWhitePrefixesMA as Pool
from trader.buyer import BaseBuyer as Buyer
from trader.seller_groups import DeepseekGroupSeller as Seller

from selector.selector_deepseek import select

data_source = DataSource.MOOTDX     # DataSource.AKSHARE 数据源也可以 TUSHARE 需要配置 token

STRATEGY_NAME = 'AI智选'  # DEEPSEEK 生成的策略，仅做展示使用，不保证收益
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
PATH_INFO = PATH_BASE + '/tmp_{}.pkl'           # 用来缓存当天的指标信息
disk_lock = threading.Lock()                    # 操作磁盘文件缓存的锁
cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重


class PoolConf:
    white_prefixes = {'00', '60', '30'}
    white_index_symbol = IndexSymbol.INDEX_ZZ_ALL
    white_ma_above_period = 10

    black_prompts = [
        'ST',
        '退市',
        '近一周大股东减持',
    ]
    day_count = 200                 # 200个足够算出周期为120的均线数据
    price_adjust = ExitRight.QFQ    # 历史价格复权
    columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']


class BuyConf:
    time_ranges = [['14:55', '14:57']]
    interval = 15
    order_premium = 0.05    # 保证成功买入成交的溢价

    slot_count = 30         # 持股数量上限
    slot_capacity = 10000   # 每个仓的资金上限
    daily_buy_max = 10      # 单日买入股票上限
    once_buy_limit = 30     # 单次选股最多买入股票数量
    inc_limit = 1.20        # 相对于昨日收盘的涨幅限制
    min_price = 3.00        # 限制最低可买入股票的现价


class SellConf:
    time_ranges = [['09:31', '11:30'], ['13:00', '14:57']]
    interval = 1
    order_premium = 0.05            # 保证成功卖出成交的溢价

    switch_time_range = ['14:30', '14:57']
    switch_hold_days = 5            # 持仓天数
    switch_demand_daily_up = 0.002  # 换仓上限乘数

    hard_time_range = ['09:31', '14:57']
    earn_limit = 9.999              # 硬性止盈率
    risk_limit = 1 - 0.03           # 硬性止损率
    risk_tight = 0.002              # 硬性止损率每日上移

    fall_time_range = ['09:31', '14:57']
    fall_from_top = [
        (1.05, 9.99, 0.02),
        (1.02, 1.05, 0.05),
    ]

    return_time_range = ['09:31', '14:57']
    return_of_profit = [
        (1.07, 9.99, 0.35),
        (1.02, 1.07, 0.95),
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
    my_suber.update_code_list(my_pool.get_code_list() + hold_list)

    # prepare_history() -> None:
    now = datetime.datetime.now()
    for i in range(15, 30):
        delete_file(PATH_INFO.format((now - datetime.timedelta(days=i)).strftime('%Y_%m_%d')))
    cache_path = PATH_INFO.format(now.strftime('%Y_%m_%d'))

    start = get_prev_trading_date(now, PoolConf.day_count)
    end = get_prev_trading_date(now, 1)

    # 白名单加持仓列表
    positions = my_delegate.check_positions()
    history_list = my_pool.get_code_list()
    history_list += [position.stock_code for position in positions if is_symbol(position.stock_code)]

    my_suber.download_cache_history(
        cache_path=cache_path,
        code_list=history_list,
        start=start,
        end=end,
        adjust=PoolConf.price_adjust,
        columns=PoolConf.columns,
        data_source=data_source
    )


def near_trade_begin():
    now = datetime.datetime.now()
    start = get_prev_trading_date(now, PoolConf.day_count)
    end = get_prev_trading_date(now, 1)

    positions = my_delegate.check_positions()
    history_list = my_pool.get_code_list()
    history_list += [position.stock_code for position in positions if is_symbol(position.stock_code)]
    # 使用 AKSHARE 数据源这些代码其实没作用
    my_suber.refresh_memory_history(code_list=history_list, start=start, end=end, data_source=data_source)


# ======== 买点 ========


def check_stock(code: str, quote: Dict, curr_date: str) -> bool:
    df = concat_ak_quote_dict(my_suber.cache_history[code], quote, curr_date)

    result_df = select(df, code, quote)
    buy = result_df['PASS'].values[-1]

    return buy


def select_stocks(quotes: dict, curr_date: str) -> dict[str, dict]:
    selections = {}

    for code in quotes:
        if code not in my_suber.cache_history:
            # print(f'{code} 没有历史数据')
            continue

        if code not in my_pool.cache_whitelist:
            # debug(code, f'不在白名单')
            continue

        if code in my_pool.cache_blacklist:
            # debug(code, f'在黑名单')
            continue

        quote = quotes[code]

        passed = check_stock(code, quote, curr_date)
        if not passed:
            # debug(f'{code} {info}')
            continue

        prev_close = quote['lastClose']
        curr_open = quote['open']
        curr_price = quote['lastPrice']

        if not curr_price > BuyConf.min_price:
            debug(code, f'价格小于{BuyConf.min_price}')
            continue

        if not curr_open <= curr_price <= prev_close * BuyConf.inc_limit:
            debug(code, f'涨幅不符合区间 {curr_open} <= {curr_price} <= {prev_close * BuyConf.inc_limit}')
            continue

        # if quote['pvolume'] > 0:
        #     average_price = quote['amount'] / quote['pvolume']
        #     if not curr_price > average_price:
        #         debug(code, f'现价小于当日成交均价')
        #         continue

        selections[code] = {
            'price': max(quote['askPrice'] + [quote['lastPrice']]),
            'lastClose': quote['lastClose'],
        }

    return selections


def scan_buy(quotes: Dict, curr_date: str, positions: List) -> None:
    selections = select_stocks(quotes, curr_date)
    debug(len(quotes), selections)

    global cache_selected
    cache_selected = my_buyer.buy_selections(selections, cache_selected, curr_date, positions)


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
        near_trade_begin=near_trade_begin,
        use_ap_scheduler=True,
        ding_messager=DING_MESSAGER,
        open_tick_memory_cache=True,
        open_today_deal_report=True,
        open_today_hold_report=True,
    )
    my_suber.start_scheduler()
