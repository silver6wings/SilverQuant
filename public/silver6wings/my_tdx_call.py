import math
import logging

from credentials import *

from tools.utils_basic import logging_init, is_symbol, get_limit_up_price
from tools.utils_cache import *
from tools.utils_ding import DingMessager

from delegate.xt_subscriber import XtSubscriber, update_position_held

from trader.pools import StocksPoolWhiteCustomTdx as Pool
from trader.buyer import BaseBuyer as Buyer


STRATEGY_NAME = '竞价TDX'  # 读通达信列表，09:20 挂涨停价买
DING_MESSAGER = DingMessager(DING_SECRET, DING_TOKENS)
IS_PROD = False     # 生产环境标志：False 表示使用掘金模拟盘 True 表示使用QMT账户下单交易
IS_DEBUG = True     # 日志输出标记：控制台是否打印debug方法的输出

PATH_BASE = CACHE_PROD_PATH if IS_PROD else CACHE_TEST_PATH

PATH_ASSETS = PATH_BASE + '/assets.csv'         # 记录历史净值
PATH_DEAL = PATH_BASE + '/deal_hist.csv'        # 记录历史成交
PATH_HELD = PATH_BASE + '/held_days.json'       # 记录持仓日期
PATH_MAXP = PATH_BASE + '/max_price.json'       # 记录建仓后历史最高
PATH_MINP = PATH_BASE + '/min_price.json'       # 记录建仓后历史最低
PATH_LOGS = PATH_BASE + '/logs.txt'             # 记录策略的历史日志
disk_lock = threading.Lock()                    # 操作磁盘文件缓存的锁
cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重
cache_history: Dict[str, pd.DataFrame] = {}     # 记录历史日线行情的信息 { code: DataFrame }


def debug(*args, **kwargs):
    if IS_DEBUG:
        print(*args, **kwargs)


class PoolConf:
    tdx_codes_filepath = r'C:\new_tdx\T0002\blocknew\ZXG.blk'
    black_prompts = []

    # 忽略监控列表
    ignore_stocks = [
        '000001.SZ',        # 深交所股票尾部加.SZ
        '600000.SH',        # 上交所股票尾部加.SH
    ]


class BuyConf:
    time_ranges = [['09:15', '09:25']]
    interval = 1            # 扫描买入间隔，60的约数：1-6, 10, 12, 15, 20, 30
    order_premium = 0.00    # 保证市价单成交的溢价，单位（元）

    slot_count = 5          # 持股数量上限
    slot_capacity = 5000    # 每个仓的资金上限
    once_buy_limit = 5      # 单次选股最多买入股票数量（若单次未买进当日不会再买这只


# ======== 盘前 ========


def before_trade_day():
    # held_increase() -> None:
    update_position_held(disk_lock, my_delegate, PATH_HELD)
    if all_held_inc(disk_lock, PATH_HELD):
        logging.warning('===== 所有持仓计数 +1 =====')
        print(f'All held stock day +1!')

    # refresh_code_list():
    if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
        return

    my_pool.refresh()
    positions = my_delegate.check_positions()
    hold_list = [position.stock_code for position in positions if is_symbol(position.stock_code)]
    full_list = my_pool.get_code_list() + hold_list
    target_list = [code for code in full_list if code not in PoolConf.ignore_stocks]

    my_suber.update_code_list(target_list)


# ======== 买点 ========


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
        last_close = round(quote['lastClose'], 3)
        selection = {
            'code': code,
            'price': get_limit_up_price(code, last_close) - 0.01,
            'lastClose': last_close,
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
    STRATEGY_NAME = STRATEGY_NAME if IS_PROD else STRATEGY_NAME + "[测]"
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
            ding_messager=DING_MESSAGER,
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

    # temp_now = datetime.datetime.now()
    # temp_date = temp_now.strftime('%Y-%m-%d')
    # temp_time = temp_now.strftime('%H:%M')
    #
    # # 定时任务启动
    # schedule.every().day.at('08:05').do(held_increase)
    # schedule.every().day.at('08:10').do(refresh_code_list)
    #
    # if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
    #     held_increase()
    #
    #     if '08:10' < temp_time < '14:57':
    #         refresh_code_list()
    #
    #     if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
    #         my_suber.subscribe_tick()  # 重启时如果在交易时间则订阅Tick
    #
    # try:
    #     print('[定时器已启动]')
    #     while True:
    #         schedule.run_pending()
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print('[手动结束进程]')
    # finally:
    #     schedule.clear()
    #     my_delegate.shutdown()
