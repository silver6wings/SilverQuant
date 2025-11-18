import logging
import redis

from credentials import *
from tools.utils_basic import logging_init
from tools.utils_cache import *
from tools.utils_ding import DingMessager

from delegate.xt_subscriber import XtSubscriber

from trader.pools import StocksPoolWhitePrefixes as Pool


STRATEGY_NAME = '数据推送'
DING_MESSAGER = DingMessager(DING_SECRET, DING_TOKENS)
IS_PROD = False     # 生产环境标志：False 表示使用掘金模拟盘 True 表示使用QMT账户下单交易
IS_DEBUG = True     # 日志输出标记：控制台是否打印debug方法的输出

REDIS_HOST = '192.168.1.6'
REDIS_PORT = 6379
REDIS_CHANNEL = 'quotes_data'

PATH_BASE = CACHE_PROD_PATH if IS_PROD else CACHE_TEST_PATH
PATH_ASSETS = ''
PATH_DEAL = ''
PATH_LOGS = PATH_BASE + '/logs.txt'             # 记录策略的历史日志
disk_lock = threading.Lock()                    # 操作磁盘文件缓存的锁
cache_selected: Dict[str, Set] = {}             # 记录选股历史，去重


class PoolConf:
    white_prefixes = {'00', '60', '30', '68'}
    white_none_st = False
    black_prompts = []


# ======== 盘前 ========


def before_trade_day() -> None:
    my_pool.refresh()
    my_suber.update_code_list(my_pool.get_code_list())


# ======== 框架 ========


def execute_strategy(curr_date: str, curr_time: str, curr_seconds: str, curr_quotes: Dict) -> bool:
    # print(curr_date, curr_time, curr_seconds, curr_quotes)
    print(f'[{len(curr_quotes)}]', end='')
    messages = {
        'curr_date': curr_date,
        'curr_time': curr_time,
        'curr_seconds': curr_seconds,
        'curr_quotes': curr_quotes,
    }
    my_redis.publish(REDIS_CHANNEL, json.dumps(messages))  # 发送消息到Redis频道
    return True


if __name__ == '__main__':
    logging_init(path=PATH_LOGS, level=logging.INFO)
    STRATEGY_NAME = STRATEGY_NAME if IS_PROD else STRATEGY_NAME + "[测]"
    print(f'正在启动 {STRATEGY_NAME}...')
    my_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    my_pool = Pool(
        account_id=QMT_ACCOUNT_ID,
        strategy_name=STRATEGY_NAME,
        parameters=PoolConf,
        ding_messager=DING_MESSAGER,
    )
    my_suber = XtSubscriber(
        account_id=QMT_ACCOUNT_ID,
        strategy_name=STRATEGY_NAME,
        delegate=None,
        path_deal=PATH_DEAL,
        path_assets=PATH_ASSETS,
        execute_strategy=execute_strategy,
        before_trade_day=before_trade_day,
        use_ap_scheduler=True,
        ding_messager=DING_MESSAGER,
        open_tick_memory_cache=False,
    )
    my_suber.start_scheduler()
