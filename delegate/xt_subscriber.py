import json
import pickle
import datetime
import time
import functools
import random

import threading
import os
import pandas as pd

from typing import Dict, Callable, Optional

from xtquant import xtdata

from delegate.xt_delegate import XtDelegate
from reader.daily_history import DailyHistoryCache

from tools.utils_basic import code_to_symbol
from tools.utils_cache import StockNames, check_is_open_day, get_total_asset_increase
from tools.utils_cache import load_pickle, save_pickle, load_json, save_json
from tools.utils_ding import DingMessager
from tools.utils_remote import DataSource, get_daily_history, quote_to_tick


def check_open_day(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return                      # 非开放日直接 return，不执行函数
        return func(*args, **kwargs)    # 开放日正常执行
    return wrapper


def colour_text(text: str, to_red: bool, to_green: bool):
    color = '#3366FF'
    # （红色RGB为：220、40、50，绿色RGB为：22、188、80）
    if to_red:
        color = '#DC2832'
    if to_green:
        color = '#16BC50'

    return f'<font color="{color}">{text}</font>'


class XtSubscriber:
    def __init__(
        self,
        account_id: str,
        strategy_name: str,
        delegate: Optional[XtDelegate],
        path_deal: str,
        path_assets: str,
        execute_strategy: Callable,         # 策略回调函数
        execute_interval: int = 1,          # 策略执行间隔，单位（秒）
        before_trade_day: Callable = None,  # 盘前函数
        finish_trade_day: Callable = None,  # 盘后函数
        use_ap_scheduler: bool = False,     # 默认使用旧版 schedule
        ding_messager: DingMessager = None,
        open_tick_memory_cache: bool = False,
        tick_memory_data_frame: bool = False,
        open_today_deal_report: bool = False,
        open_today_hold_report: bool = False,
    ):
        self.account_id = '**' + str(account_id)[-4:]
        self.strategy_name = strategy_name
        self.delegate = delegate

        self.path_deal = path_deal
        self.path_assets = path_assets

        self.execute_strategy = execute_strategy
        self.execute_interval = execute_interval
        self.before_trade_day = before_trade_day
        self.finish_trade_day = finish_trade_day
        self.ding_messager = ding_messager

        self.lock_quotes_update = threading.Lock()  # 聚合实时打点缓存的锁

        self.cache_quotes: Dict[str, Dict] = {}     # 记录实时的价格信息
        self.cache_limits: Dict[str, str] = {       # 限制执行次数的缓存集合
            'prev_seconds': '',                     # 限制每秒一次跑策略扫描的缓存
            'prev_minutes': '',                     # 限制每分钟屏幕心跳换行的缓存
        }
        self.cache_history: Dict[str, pd.DataFrame] = {}    # 记录历史日线行情的信息 { code: DataFrame }

        self.open_tick = open_tick_memory_cache
        self.is_ticks_df = tick_memory_data_frame
        self.quick_ticks: bool = False                          # 是否开启quick tick模式
        self.today_ticks: Dict[str, list | pd.DataFrame] = {}   # 记录tick的历史信息

        self.open_today_deal_report = open_today_deal_report
        self.open_today_hold_report = open_today_hold_report

        self.code_list = ['000001.SH']  # 默认只有上证指数
        self.stock_names = StockNames()
        self.last_callback_time = datetime.datetime.now()

        self.use_ap_scheduler = use_ap_scheduler
        if self.use_ap_scheduler:
            from apscheduler.schedulers.blocking import BlockingScheduler
            self.scheduler = BlockingScheduler()

    # -----------------------
    # 策略触发主函数
    # -----------------------
    def callback_sub_whole(self, quotes: Dict) -> None:
        now = datetime.datetime.now()
        self.last_callback_time = now

        curr_date = now.strftime('%Y-%m-%d')
        curr_time = now.strftime('%H:%M')

        # 每分钟输出一行开头
        if self.cache_limits['prev_minutes'] != curr_time:
            self.cache_limits['prev_minutes'] = curr_time
            print(f'\n[{curr_time}]', end='')

        curr_seconds = now.strftime('%S')
        with self.lock_quotes_update:
            self.cache_quotes.update(quotes)  # 合并最新数据

        if self.open_tick and (not self.quick_ticks):
            self.record_tick_to_memory(quotes)  # 更全（默认：先记录再执行）

        # 执行策略
        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            if int(curr_seconds) % self.execute_interval == 0:
                print('.' if len(self.cache_quotes) > 0 else 'x', end='')  # 每秒钟开始的时候输出一个点

                if self.execute_strategy(
                    curr_date,
                    curr_time,
                    curr_seconds,
                    self.cache_quotes,
                ):
                    with self.lock_quotes_update:
                        if self.open_tick and self.quick_ticks:
                            self.record_tick_to_memory(self.cache_quotes)  # 更快（先执行再记录）
                        self.cache_quotes.clear()  # execute_strategy() return True means need clear

    # -----------------------
    # 监测主策略执行
    # -----------------------
    def callback_monitor(self):
        now = datetime.datetime.now()

        if not check_is_open_day(now.strftime('%Y-%m-%d')):
            return

        if now - self.last_callback_time > datetime.timedelta(minutes=1):
            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name}:中断\n请检查QMT数据源 ',
                    alert=True,
                )
            if len(self.code_list) > 1 and xtdata.get_client():
                print('尝试重新订阅行情数据')
                time.sleep(1)
                self.resubscribe_tick(notice=True)

    # -----------------------
    # 订阅tick相关
    # -----------------------
    def subscribe_tick(self, resume: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                               f'{"恢复" if resume else "启动"} {len(self.code_list) - 1}支')
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        xtdata.enable_hello = False
        print('[启动行情订阅]', end='')

    def unsubscribe_tick(self, pause: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if 'sub_seq' in self.cache_limits:
            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                   f'{"暂停" if pause else "关闭"}')
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
            print('\n[关闭行情订阅]')

    def resubscribe_tick(self, notice: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if 'sub_seq' in self.cache_limits:
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        xtdata.enable_hello = False

        if self.ding_messager is not None and notice:
            self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                               f'重启 {len(self.code_list) - 1}支')
        print('\n[重启行情订阅]', end='')
    
    def update_code_list(self, code_list: list[str]):
        # 加上证指数防止没数据不打点
        self.code_list = ['000001.SH'] + code_list

    # -----------------------
    # 盘中实时的tick历史
    # -----------------------
    def record_tick_to_memory(self, quotes):
        # 记录 tick 历史
        if self.is_ticks_df:
            tick_df_cols = ['time', 'price', 'volume', 'amount'] \
                + [f'askPrice{i}' for i in range(1, 6)] \
                + [f'askVol{i}' for i in range(1, 6)] \
                + [f'bidPrice{i}' for i in range(1, 6)] \
                + [f'bidVol{i}' for i in range(1, 6)]
            for code in quotes:
                if code not in self.today_ticks:
                    self.today_ticks[code] = pd.DataFrame(columns=tick_df_cols)
                quote = quotes[code]
                tick = quote_to_tick(quote)
                df = self.today_ticks[code]
                df.loc[len(df)] = tick.values()     # 加入最后一行
        else:
            for code in quotes:
                if code not in self.today_ticks:
                    self.today_ticks[code] = []

                quote = quotes[code]
                tick_time = datetime.datetime.fromtimestamp(quote['time'] / 1000).strftime('%H:%M:%S')
                self.today_ticks[code].append([
                    tick_time,                          # 成交时间，格式：%H:%M:%S
                    round(quote['lastPrice'], 3),       # 成交价格
                    int(quote['volume']),               # 累计成交量（手）
                    round(quote['amount'], 3),          # 累计成交额（元）
                    [round(p, 3) if isinstance(p, (int, float)) else p for p in quote['askPrice']],  # 卖价
                    [int(v) if isinstance(v, (int, float)) else v for v in quote['askVol']],         # 卖量
                    [round(p, 3) if isinstance(p, (int, float)) else p for p in quote['bidPrice']],  # 买价
                    [int(v) if isinstance(v, (int, float)) else v for v in quote['bidVol']],         # 买量
                ])

    def clean_ticks_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        self.today_ticks.clear()
        self.today_ticks = {}
        print(f"已清除tick缓存")

    def save_tick_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.is_ticks_df:
            pickle_file = './_cache/debug/tick_history.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.today_ticks, f)
            print(f"当日tick数据已存储为 {pickle_file} 文件")
        else:
            json_file = './_cache/debug/tick_history.json'
            with open(json_file, 'w') as file:
                json.dump(self.today_ticks, file, indent=4)
            print(f"当日tick数据已存储为 {json_file} 文件")

    # -----------------------
    # 盘前下载数据缓存
    # -----------------------
    def download_from_remote(
        self,
        target_codes: list,
        start: str,
        end: str,
        adjust: str,
        columns: list[str],
        data_source: int,
    ):
        print(f'Prepared time range: {start} - {end}')
        t0 = datetime.datetime.now()

        print(f'Downloading {len(target_codes)} stocks from {start} to {end} ...')
        group_size = 200
        for i in range(0, len(target_codes), group_size):
            sub_codes = [sub_code for sub_code in target_codes[i:i + group_size]]
            time.sleep(1)
            print(i, sub_codes)  # 已更新数量

            # TUSHARE 批量下载限制总共8000天条数据，所以暂时弃用
            # if data_source == DataSource.TUSHARE:
            #     # 使用 TUSHARE 数据源批量下载
            #     dfs = get_ts_daily_histories(sub_codes, start, end, columns)
            #     self.cache_history.update(dfs)
            #     time.sleep(0.1)

            # 默认使用 AKSHARE 数据源
            for code in sub_codes:
                df = get_daily_history(code, start, end, columns=columns, adjust=adjust, data_source=data_source)
                if df is not None:
                    self.cache_history[code] = df
                if data_source == DataSource.TUSHARE:
                    time.sleep(0.1)

        t1 = datetime.datetime.now()
        print(f'Prepared TIME COST: {t1 - t0}')

    def download_cache_history(
        self,
        cache_path: str,  # DATA SOURCE 是tushare的时候不需要
        code_list: list[str],
        start: str,
        end: str,
        adjust: str,
        columns: list[str],
        data_source: int = DataSource.AKSHARE,
    ):
        if data_source == DataSource.AKSHARE:
            temp_indicators = load_pickle(cache_path)
            if temp_indicators is not None and len(temp_indicators) > 0:
                # 如果有缓存就读缓存
                self.cache_history.clear()
                self.cache_history = {}
                self.cache_history.update(temp_indicators)
                print(f'{len(self.cache_history)} histories loaded from {cache_path}')
                if self.ding_messager is not None:
                    self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                       f'加载{len(self.cache_history)}支')
            else:
                # 如果没缓存就刷新白名单
                self.cache_history.clear()
                self.cache_history = {}
                self.download_from_remote(code_list, start, end, adjust, columns, data_source)
                save_pickle(cache_path, self.cache_history)
                print(f'{len(self.cache_history)} of {len(code_list)} histories saved to {cache_path}')
                if self.ding_messager is not None:
                    self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                       f'下载{len(self.cache_history)}支')
        elif data_source == DataSource.TUSHARE:
            hc = DailyHistoryCache()
            hc.daily_history.download_recent_daily(5)

            # 计算两个日期之间的差值
            start_date = datetime.datetime.strptime(start, '%Y%m%d')
            end_date = datetime.datetime.strptime(end, '%Y%m%d')
            delta = abs(end_date - start_date)

            self.cache_history = hc.daily_history.get_subset_copy(None, delta.days + 1)
            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                   f'更新{len(self.cache_history)}支')
        else:
            if self.ding_messager is not None:
                self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                   f'无法识别数据源')

    # -----------------------
    # 盘后报告总结
    # -----------------------
    def daily_summary(self):
        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if not check_is_open_day(curr_date):
            return

        if self.open_today_deal_report:
            self.today_deal_report(today=curr_date)

        if self.delegate is not None:
            if self.open_today_hold_report:
                positions = self.delegate.check_positions()
                self.today_hold_report(today=curr_date, positions=positions)

            asset = self.delegate.check_asset()
            self.check_asset(today=curr_date, asset=asset)

    def today_deal_report(self, today):
        if not os.path.exists(self.path_deal):
            print('Missing deal record file!')
            return

        df = pd.read_csv(self.path_deal, encoding='gbk')
        if '日期' in df.columns:
            df = df[df['日期'] == today]

        title = f'[{self.account_id}]{self.strategy_name} 委托统计'
        text = f'{title}\n\n[{today}] 交易{len(df)}单'

        if len(df) > 0:
            for index, row in df.iterrows():
                # ['日期', '时间', '代码', '名称', '类型', '注释', '成交价', '成交量']
                text += '\n\n> '
                text += f'{row["时间"]} {row["注释"]} {code_to_symbol(row["代码"])} '
                text += '\n>\n> '
                text += f'{row["名称"]} {row["成交量"]}股 {row["成交价"]}元 '

        if self.ding_messager is not None:
            self.ding_messager.send_markdown(title, text)

    def today_hold_report(self, today, positions):
        text = ''
        i = 0
        for position in positions:
            if position.volume > 0:
                code = position.stock_code
                quotes = xtdata.get_full_tick([code])
                curr_price = None
                if (code in quotes) and ('lastPrice' in quotes[code]):
                    curr_price = quotes[code]['lastPrice']

                open_price = position.open_price
                if open_price == 0.0 or curr_price is None:
                    continue

                vol = position.volume

                i += 1
                text += '\n\n>'
                text += f'' \
                        f'{code_to_symbol(code)} ' \
                        f'{self.stock_names.get_name(code)} ' \
                        f'{curr_price * vol:.2f}元'
                text += '\n>\n>'

                total_change = colour_text(
                    f"{(curr_price - open_price) * vol:.2f}",
                    curr_price > open_price,
                    curr_price < open_price,
                )
                ratio_change = colour_text(
                    f'{(curr_price / open_price - 1) * 100:.2f}%',
                    curr_price > open_price,
                    curr_price < open_price,
                )

                text += f'盈亏比:{ratio_change}</font> 盈亏额:{total_change}</font>'

        title = f'[{self.account_id}]{self.strategy_name} 持仓统计'
        text = f'{title}\n\n[{today}] 持仓{i}支\n{text}'

        if self.ding_messager is not None:
            self.ding_messager.send_markdown(title, text)

    def check_asset(self, today, asset):
        title = f'[{self.account_id}]{self.strategy_name} 盘后清点'
        text = title

        increase = get_total_asset_increase(self.path_assets, today, asset.total_asset)
        if increase is not None:
            text += '\n>\n> '

            total_change = colour_text(
                f'{"+" if increase > 0 else ""}{round(increase, 2)}',
                increase > 0,
                increase < 0,
            )
            ratio_change = colour_text(
                f'{"+" if increase > 0 else ""}{round(increase * 100 / asset.total_asset, 2)}%',
                increase > 0,
                increase < 0,
            )
            text += f'当日变动: {total_change}元({ratio_change})'

        text += '\n>\n> '
        text += f'持仓市值: {round(asset.market_value, 2)}元'

        text += '\n>\n> '
        text += f'剩余现金: {round(asset.cash, 2)}元'

        text += f'\n>\n>'
        text += f'资产总计: {round(asset.total_asset, 2)}元'

        if self.ding_messager is not None:
            self.ding_messager.send_markdown(title, text)

    # -----------------------
    # 定时器
    # -----------------------
    @check_open_day
    def before_trade_day_wrapper(self):
        if self.before_trade_day is not None:
            self.before_trade_day()

    @check_open_day
    def finish_trade_day_wrapper(self):
        if self.finish_trade_day is not None:
            self.finish_trade_day()

    def start_scheduler(self):
        # 默认定时任务列表
        cron_jobs = [
            ['08:00', prev_check_open_day, None],
            ['09:15', self.subscribe_tick, None],
            ['11:30', self.unsubscribe_tick, (True,)],
            ['13:00', self.subscribe_tick, (True,)],
            ['15:00', self.unsubscribe_tick, None],
            ['15:01', self.daily_summary, None],
        ]

        if self.before_trade_day is not None:
            cron_jobs.append([f'08:{random.randint(0, 25) + 5}', self.before_trade_day_wrapper, None])

        if self.finish_trade_day is not None:
            cron_jobs.append([f'16:{random.randint(0, 10) + 5}', self.finish_trade_day_wrapper, None])

        if self.open_tick:
            cron_jobs.append(['09:10', self.clean_ticks_history, None])
            cron_jobs.append(['15:10', self.save_tick_history, None])

        # 数据源中断检查时间点
        monitor_time_list = [
            '09:35', '09:45', '09:55', '10:05', '10:15', '10:25',
            '10:35', '10:45', '10:55', '11:05', '11:15', '11:25',
            '13:05', '13:15', '13:25', '13:35', '13:45', '13:55',
            '14:05', '14:15', '14:25', '14:35', '14:45', '14:55',
        ]

        temp_now = datetime.datetime.now()
        temp_date = temp_now.strftime('%Y-%m-%d')
        temp_time = temp_now.strftime('%H:%M')

        if self.use_ap_scheduler:
            # 新版 apscheduler
            for cron_job in cron_jobs:
                [hr, mn] = cron_job[0].split(':')
                if cron_job[2] is None:
                    self.scheduler.add_job(cron_job[1], 'cron', hour=hr, minute=mn)
                else:
                    self.scheduler.add_job(cron_job[1], 'cron', hour=hr, minute=mn, args=list(cron_job[2]))

            # 尝试重新订阅 tick 数据，减少30分时无数据返回机率
            self.scheduler.add_job(self.resubscribe_tick, 'cron', hour=9, minute=29, second=30) 

            for monitor_time in monitor_time_list:
                [hr, mn] = monitor_time.split(':')
                self.scheduler.add_job(self.callback_monitor, 'cron', hour=hr, minute=mn)

            # 盘中执行需要补齐
            if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
                self.before_trade_day_wrapper()
                if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
                    self.subscribe_tick()  # 重启时如果在交易时间则订阅Tick

            # 启动定时器
            try:
                print('策略定时器任务已经启动！')
                self.scheduler.start()
            except KeyboardInterrupt:
                print('手动结束进程，请检查缓存文件是否因读写中断导致空文件错误')
            except Exception as e:
                print('策略定时器出错：', e)
            finally:
                self.delegate.shutdown()
        else:
            # 旧版 schedule
            import schedule
            for cron_job in cron_jobs:
                if cron_job[2] is None:
                    schedule.every().day.at(cron_job[0]).do(cron_job[1])
                else:
                    schedule.every().day.at(cron_job[0]).do(cron_job[1], list(cron_job[2])[0])

            for monitor_time in monitor_time_list:
                schedule.every().day.at(monitor_time).do(self.callback_monitor)

            # 盘中执行需要补齐，旧代码都放在策略文件里了这里就不重复执行破坏老代码
            # if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
            #     self._before_trade_day()
            #     if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
            #         self.subscribe_tick()  # 重启时如果在交易时间则订阅Tick

            # 旧代码还有别的要执行，没有放在 before_trade_day 所以这里虽然不优雅单也先注释掉
            # try:
            #     while True:
            #         schedule.run_pending()
            #         time.sleep(1)
            # except KeyboardInterrupt:
            #     print('手动结束进程，请检查缓存文件是否因读写中断导致空文件错误')
            # finally:
            #     schedule.clear()
            #     self.delegate.shutdown()


# -----------------------
# 检查是否交易日
# -----------------------
def prev_check_open_day():
    now = datetime.datetime.now()
    curr_date = now.strftime('%Y-%m-%d')
    curr_time = now.strftime('%H:%M')
    print(f'[{curr_time}]', end='')
    check_is_open_day(curr_date)


# -----------------------
# 持仓自动发现
# -----------------------
def update_position_held(lock: threading.Lock, delegate: XtDelegate, path: str):
    with lock:
        positions = delegate.check_positions()

        held_days = load_json(path)

        # 添加未被缓存记录的持仓
        for position in positions:
            if position.can_use_volume > 0:
                if position.stock_code not in held_days.keys():
                    held_days[position.stock_code] = 0

        if positions is not None and len(positions) > 0:
            # 删除已清仓的held_days记录
            position_codes = [position.stock_code for position in positions]
            print('当前持仓：', position_codes)
            holding_codes = list(held_days.keys())
            for code in holding_codes:
                if len(code) > 0 and code[0] != '_':
                    if code not in position_codes:
                        del held_days[code]
        else:
            print('当前空仓！')

        save_json(path, held_days)


# -----------------------
# 订阅单个股票历史N个分钟级K线
# -----------------------
def sub_quote(
    callback: Callable,
    code: str,
    count: int = -1,
    period: str = '1m',
):
    xtdata.subscribe_quote(code, period=period, count=count, callback=callback)
