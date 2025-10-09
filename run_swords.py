import logging

from credentials import *

from tools.utils_basic import logging_init, is_symbol, time_diff_seconds
from tools.utils_cache import *
from tools.utils_ding import DingMessager

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolWhiteCustomSymbol as Pool
from trader.buyer import BaseBuyer as Buyer


STRATEGY_NAME = '进攻监控'
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
    white_codes_filepath = './_cache/_pool_whitelist.txt'
    black_prompts = []

    # 忽略监控列表
    ignore_stocks = [
        '000001.SZ',        # 深交所股票尾部加.SZ
        '600000.SH',        # 上交所股票尾部加.SH
    ]


class BuyConf:
    time_ranges = [['09:31', '11:30'], ['13:00', '14:57']]
    interval = 1            # 扫描买入间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.00    # 保证市价单成交的溢价，单位（元）

    slot_count = 5          # 持股数量上限
    slot_capacity = 5000    # 每个仓的资金上限
    daily_buy_max = 10      # 单日买入股票上限
    once_buy_limit = 5      # 单次选股最多买入股票数量

    # 开始时间，结束时间，封板量突破点，封板额突破点
    # start_time, end_time, block_volume, block_amount
    blocks = [
        ['09:31', '11:30', 1000, 10000],
        ['13:01', '14:55', 1000, 10000],
    ]

    # 封板秒数要求
    block_seconds = 3


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


# ======== 买点 ========


def check_is_blocking(quote: dict, curr_time: str) -> (bool, int, float):
    ask_vol = quote['askVol'][0]
    bid_vol = quote['bidVol'][0]
    curr_price = quote['lastPrice']
    bid_amount = bid_vol * curr_price

    is_limiting_up = ask_vol < 0.001
    if not is_limiting_up:
        return False, 0, 0

    for block in BuyConf.blocks:
        if block[0] <= curr_time < block[1]:
            block_volume = block[2]
            block_amount = block[3]

            is_volume_stab = bid_vol > block_volume
            is_amount_stab = bid_amount > block_amount
            if is_volume_stab and is_amount_stab:
                return True, bid_vol, bid_amount

    return False, bid_vol, bid_amount


def check_block_ticks(
    quote: dict,
    curr_time: str,
    curr_seconds: str,
    ticks: list,
) -> bool:
    curr_price = quote['lastPrice']

    # 获取一段之前的最大价格
    max_price = curr_price
    i = len(ticks) - 1
    now_seconds = datetime.datetime.strptime(f'{curr_time}:{curr_seconds}', '%H:%M:%S')
    while i >= 0:
        i_seconds = datetime.datetime.strptime(ticks[i][0], '%H:%M:%S')
        if time_diff_seconds(now_seconds, i_seconds) > BuyConf.block_seconds:
            break
        i_price = ticks[i][1]
        if i_price < max_price:  # 保证时间段内的价格都是最大
            return False

        max_price = max(max_price, i_price)
        i -= 1

    # 最后还要保证没有已经炸板的情况
    return curr_price >= max_price


def select_stocks(
    quotes: Dict,
    curr_time: str,
    curr_seconds: str,
) -> dict[str, dict]:
    selections = {}

    for code in quotes:
        if code not in my_pool.cache_whitelist:
            # debug(code, f'不在白名单')
            continue

        if code in my_pool.cache_blacklist:
            # debug(code, f'在黑名单')
            continue

        quote = quotes[code]

        # 是否涨停封住
        limiting_up, bid_vol, bid_amt = check_is_blocking(quote, curr_time)
        if not limiting_up:
            # if bid_vol > 0 and bid_amt > 0:
            #     debug(code, f'封单量{bid_vol} 封单额{bid_amt}')
            continue

        # 检查封板时间足够
        block_enough = check_block_ticks(quote, curr_time, curr_seconds, my_suber.today_ticks[code])
        if not block_enough:
            continue

        selections[code] = {
            'price': max(quote['askPrice'] + [quote['lastPrice']]),
            'lastClose': quote['lastClose'],
        }

    return selections


def scan_buy(
    quotes: Dict,
    curr_date: str,
    curr_time: str,
    curr_seconds: str,
    positions: List,
) -> None:
    selections = select_stocks(quotes, curr_time, curr_seconds)
    # debug(f'本次扫描:{len(quotes)}, 选股{selections})

    global cache_selected
    cache_selected = my_buyer.buy_selections(selections, cache_selected, curr_date, positions)


# ======== 框架 ========


def execute_strategy(curr_date: str, curr_time: str, curr_seconds: str, curr_quotes: Dict) -> bool:
    positions = my_delegate.check_positions()

    for time_range in BuyConf.time_ranges:
        if time_range[0] <= curr_time <= time_range[1]:
            if int(curr_seconds) % BuyConf.interval == 0:
                scan_buy(curr_quotes, curr_date, curr_time, curr_seconds, positions)
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
