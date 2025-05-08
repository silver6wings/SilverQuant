import time
import math
import logging
import schedule

from credentials import *

from tools.utils_basic import logging_init, is_symbol, time_diff_seconds
from tools.utils_cache import *
from tools.utils_ding import DingMessager

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.buyer import BaseBuyer as Buyer
from trader.pools import StocksPoolWhiteCustomSymbol as Pool
from trader.seller_groups import ShieldGroupSeller as Seller


STRATEGY_NAME = '进攻监控'
DING_MESSAGER = DingMessager(DING_SECRET, DING_TOKENS)
IS_PROD = True
IS_DEBUG = True

PATH_BASE = CACHE_BASE_PATH

PATH_ASSETS = PATH_BASE + '/assets.csv'         # 记录历史净值
PATH_DEAL = PATH_BASE + '/deal_hist.csv'        # 记录历史成交
PATH_HELD = PATH_BASE + '/held_days.json'       # 记录持仓日期
PATH_MAXP = PATH_BASE + '/max_price.json'       # 记录建仓后历史最高
PATH_MINP = PATH_BASE + '/min_price.json'       # 记录建仓后历史最低
PATH_LOGS = PATH_BASE + '/logs.txt'             # 用来存储选股和委托操作

disk_lock = threading.Lock()           # 操作磁盘文件缓存的锁

cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重
cache_history: Dict[str, pd.DataFrame] = {}     # 记录历史日线行情的信息 { code: DataFrame }


def debug(*args):
    if IS_DEBUG:
        print(*args)


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
    once_buy_limit = 5      # 单次选股最多买入股票数量（若单次未买进当日不会再买这只

    # 开始时间，结束时间，封板量突破点，封板额突破点
    # start_time, end_time, block_volume, block_amount
    blocks = [
        ['09:31', '11:30', 1000, 10000],
        ['13:01', '14:55', 1000, 10000],
    ]

    # 封板秒数要求
    block_seconds = 3


class SellConf:
    time_ranges = [['09:31', '11:30'], ['13:00', '14:57']]
    interval = 1                    # 扫描买入间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.03            # 保证市价单成交的溢价，单位（元）


# ======== 盘前 ========


def held_increase() -> None:
    if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
        return

    update_position_held(disk_lock, my_delegate, PATH_HELD)
    if all_held_inc(disk_lock, PATH_HELD):
        logging.warning('===== 所有持仓计数 +1 =====')
        print(f'All held stock day +1!')


def refresh_code_list():
    if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
        return

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
    curr_date: str,
    curr_time: str,
    curr_seconds: str,
) -> List[Dict[str, any]]:
    selections = []

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

        selection = {
            'code': code,
            'price': round(max(quote['askPrice'] + [quote['lastPrice']]), 3),
            'lastClose': round(quote['lastClose'], 3),
            'bidVol': bid_vol,
            'curr_date': curr_date,
        }
        selections.append(selection)

    return selections


def scan_buy(
    quotes: Dict,
    curr_date: str,
    curr_time: str,
    curr_seconds: str,
    positions: List,
) -> None:
    selections = select_stocks(quotes, curr_date, curr_time, curr_seconds)
    # debug(f'本次扫描:{len(quotes)}, 选股{selections})

    # 选出一个以上的股票
    if len(selections) > 0:
        position_codes = [position.stock_code for position in positions]
        position_count = get_holding_position_count(positions)
        available_cash = my_delegate.check_asset().cash
        available_slot = available_cash // BuyConf.slot_capacity

        buy_count = max(0, BuyConf.slot_count - position_count)     # 确认剩余的仓位
        buy_count = min(buy_count, available_slot)                  # 确认现金够用
        buy_count = min(buy_count, len(selections))                 # 确认选出的股票够用
        buy_count = min(buy_count, BuyConf.once_buy_limit)          # 限制一秒内下单数量
        buy_count = int(buy_count)

        for i in range(len(selections)):  # 依次买入
            # logging.info(f'买数相关：持仓{position_count} 现金{available_cash} 已选{len(selections)}')
            if buy_count > 0:
                code = selections[i]['code']
                price = selections[i]['price']
                last_close = selections[i]['lastClose']
                buy_volume = math.floor(BuyConf.slot_capacity / price / 100) * 100

                if buy_volume <= 0:
                    debug(f'{code} 不够一手')
                elif code in position_codes:
                    debug(f'{code} 正在持仓')
                elif curr_date in cache_selected and code in cache_selected[curr_date]:
                    debug(f'{code} 今日已选')
                else:
                    buy_count = buy_count - 1
                    # 如果今天未被选股过 and 目前没有持仓则记录（意味着不会加仓
                    my_buyer.order_buy(code=code, price=price, last_close=last_close,
                                       volume=buy_volume, remark='买入委托')
            else:
                break

    # 记录选股历史
    if curr_date not in cache_selected:
        cache_selected[curr_date] = set()

    for selection in selections:
        if selection['code'] not in cache_selected[curr_date]:
            cache_selected[curr_date].add(selection['code'])
            logging.warning(f"记录选股 {selection['code']}\t现价: {selection['price']:.2f}")


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
    STRATEGY_NAME = STRATEGY_NAME if IS_PROD else STRATEGY_NAME + "(模拟)"
    print(f'正在启动 {STRATEGY_NAME}...')
    if IS_PROD:
        from delegate.xt_callback import XtCustomCallback
        from delegate.xt_delegate import XtDelegate, get_holding_position_count

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
        from delegate.gm_delegate import GmDelegate, get_holding_position_count

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
        account_id=QMT_ACCOUNT_ID,
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
        ding_messager=DING_MESSAGER,
    )
    my_suber.start_scheduler()

    temp_now = datetime.datetime.now()
    temp_date = temp_now.strftime('%Y-%m-%d')
    temp_time = temp_now.strftime('%H:%M')

    # 定时任务启动
    schedule.every().day.at('08:05').do(held_increase)
    schedule.every().day.at('08:10').do(refresh_code_list)

    if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
        held_increase()

        if '08:10' < temp_time < '14:57':
            refresh_code_list()

        if '09:25' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
            my_suber.subscribe_tick()  # 重启时如果在交易时间则订阅Tick

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print('手动结束进程，请检查缓存文件是否因读写中断导致空文件错误')
    finally:
        schedule.clear()
        my_delegate.shutdown()
