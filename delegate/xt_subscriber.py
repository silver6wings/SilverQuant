import time
import datetime
import json
import pickle
import random
import threading
import traceback
from typing import Dict, Callable, Optional

import pandas as pd
from xtquant import xtdata

from delegate.xt_delegate import XtDelegate
from delegate.daily_history import DailyHistoryCache
from delegate.daily_reporter import DailyReporter

from tools.utils_cache import StockNames, InfoItem, check_is_open_day, check_open_day
from tools.utils_cache import load_pickle, save_pickle, load_json, save_json
from tools.utils_ding import BaseMessager
from tools.utils_remote import DataSource, ExitRight, get_daily_history, qmt_quote_to_tick


class BaseSubscriber:
    pass


class XtSubscriber(BaseSubscriber):
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
        near_trade_begin: Callable = None,  # 盘后函数
        finish_trade_day: Callable = None,  # 盘后函数
        use_outside_data: bool = False,     # 默认使用原版 QMT data （定期 call 数据但不传入quotes）
        use_ap_scheduler: bool = False,     # 默认使用旧版 schedule （尽可能向前兼容旧策略吧）
        ding_messager: BaseMessager = None,
        open_tick_memory_cache: bool = False,
        tick_memory_data_frame: bool = False,
        open_today_deal_report: bool = False,   # 每日交易记录报告
        open_today_hold_report: bool = False,   # 每日持仓记录报告
        today_report_show_bank: bool = False,   # 是否显示银行流水（国金QMT会卡死所以默认关闭）
    ):
        self.account_id = '**' + str(account_id)[-4:]
        self.strategy_name = strategy_name
        self.delegate = delegate
        if self.delegate is not None:
            self.delegate.subscriber = self

        self.path_deal = path_deal
        self.path_assets = path_assets

        self.execute_strategy = execute_strategy
        self.execute_interval = execute_interval
        self.before_trade_day = before_trade_day    # 提前准备某些耗时长的任务
        self.near_trade_begin = near_trade_begin    # 有些数据临近开盘才更新，这里保证内存里的数据正确
        self.finish_trade_day = finish_trade_day    # 盘后及时做一些总结汇报入库类的整理工作
        self.messager = ding_messager

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
        self.today_report_show_bank = today_report_show_bank

        self.code_list = ['000001.SH']  # 默认只有上证指数
        self.stock_names = StockNames()
        self.last_callback_time = datetime.datetime.now()
        self.__extend_codes = ['399001.SZ', '510230.SH', '512680.SH', '159915.SZ', '510500.SH',
                               '588000.SH', '159101.SZ', '399006.SZ', '159315.SZ']

        self.use_outside_data = use_outside_data
        self.use_ap_scheduler = use_ap_scheduler
        if self.use_outside_data:
            self.use_ap_scheduler = True  # 如果use_outside_data 被设置为True，则需强制使用apscheduler

        self.daily_reporter = DailyReporter(
            self.account_id,
            self.strategy_name,
            self.delegate,
            self.path_deal,
            self.path_assets,
            self.messager,
            self.use_outside_data,
            self.today_report_show_bank,
        )

        if self.use_ap_scheduler:
            from apscheduler.schedulers.blocking import BlockingScheduler
            self.scheduler = BlockingScheduler()

        if self.is_ticks_df:
            self.tick_df_cols = ['time', 'price', 'high', 'low', 'volume', 'amount'] \
                + [f'askPrice{i}' for i in range(1, 6)] \
                + [f'askVol{i}' for i in range(1, 6)] \
                + [f'bidPrice{i}' for i in range(1, 6)] \
                + [f'bidVol{i}' for i in range(1, 6)]

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

        # 执行策略
        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            print_mark = '.' if len(self.cache_quotes) > 0 else 'x'

            if int(curr_seconds) % self.execute_interval == 0:
                # 更全（默认：先记录再执行）
                if self.open_tick and (not self.quick_ticks):
                    self.record_tick_to_memory(quotes)

                if self.execute_strategy(
                    curr_date,      # str(%Y-%m-%d)
                    curr_time,      # str(%H:%M)
                    curr_seconds,   # str(%S)
                    self.cache_quotes,
                ):
                    with self.lock_quotes_update:
                        self.cache_quotes.clear()  # execute_strategy() return True means need clear

                # 更快（先执行再记录）
                if self.open_tick and self.quick_ticks:
                    self.record_tick_to_memory(quotes)

                print(print_mark, end='')  # 每秒钟开始的时候输出一个点

    @check_open_day
    def callback_run_no_quotes(self) -> None:
        now = datetime.datetime.now()
        self.last_callback_time = now

        curr_date = now.strftime('%Y-%m-%d')
        curr_time = now.strftime('%H:%M')

        # 每分钟输出一行开头
        if self.cache_limits['prev_minutes'] != curr_time:
            self.cache_limits['prev_minutes'] = curr_time
            print(f'\n[{curr_time}]', end='')

        curr_seconds = now.strftime('%S')
        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            if int(curr_seconds) % self.execute_interval == 0:
                print('.' if len(self.cache_quotes) > 0 else 'x', end='')  # 每秒钟开始的时候输出一个点

                self.execute_strategy(
                    curr_date,  # str(%Y-%m-%d)
                    curr_time,  # str(%H:%M)
                    curr_seconds,  # str(%S)
                    {},
                )

    @check_open_day
    def callback_open_no_quotes(self) -> None:
        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:开启')
        print('[启动策略]', end='')

    @check_open_day
    def callback_close_no_quotes(self) -> None:
        print('\n[关闭策略]')
        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:结束')

    # -----------------------
    # 监测主策略执行
    # -----------------------
    @check_open_day
    def callback_monitor(self):
        now = datetime.datetime.now()

        if now - self.last_callback_time > datetime.timedelta(minutes=1):
            if self.messager is not None:
                self.messager.send_text_as_md(
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
    @check_open_day
    def subscribe_tick(self, resume: bool = False):
        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                          f'{"恢复" if resume else "开启"} {len(self.code_list)}支')
        print('[开启行情订阅]', end='')
        xtdata.enable_hello = False
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)

    @check_open_day
    def unsubscribe_tick(self, pause: bool = False):
        if 'sub_seq' in self.cache_limits:
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
            print('\n[结束行情订阅]')
            if self.messager is not None:
                self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                              f'{"暂停" if pause else "关闭"}')

    @check_open_day
    def resubscribe_tick(self, notice: bool = False):
        if 'sub_seq' in self.cache_limits:
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        xtdata.enable_hello = False

        if self.messager is not None and notice:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                          f'重启 {len(self.code_list)}支')
        print('\n[重启行情订阅]', end='')

    def update_code_list(self, code_list: list[str]):
        # 加上证指数防止没数据不打点
        self.code_list = ['000001.SH'] + code_list
        extend = 10 - len(self.code_list)
        if extend > 0:
            self.code_list.extend(self.__extend_codes[:extend])  # 防止数据太少长时间不返回数据导致断流

    # -----------------------
    # 盘中实时的tick历史
    # -----------------------
    def record_tick_to_memory(self, quotes):
        # 记录 tick 历史
        if self.is_ticks_df:
            for code in quotes:
                if code not in self.today_ticks:
                    self.today_ticks[code] = pd.DataFrame(columns=self.tick_df_cols)
                quote = quotes[code]
                tick = qmt_quote_to_tick(quote)
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
                    round(quote['high'], 3),            # 成交最高价
                    round(quote['low'], 3),             # 成交最最低价
                    int(quote['volume']),               # 累计成交量（手）
                    round(quote['amount'], 3),          # 累计成交额（元）
                    [round(p, 3) if isinstance(p, (int, float)) else p for p in quote['askPrice']],  # 卖价
                    [int(v) if isinstance(v, (int, float)) else v for v in quote['askVol']],         # 卖量
                    [round(p, 3) if isinstance(p, (int, float)) else p for p in quote['bidPrice']],  # 买价
                    [int(v) if isinstance(v, (int, float)) else v for v in quote['bidVol']],         # 买量
                ])
    
    @check_open_day
    def clean_ticks_history(self):
        self.today_ticks.clear()
        self.today_ticks = {}
        print(f"已清除tick缓存")

    @check_open_day
    def save_tick_history(self):
        if self.is_ticks_df:
            pickle_file = f'./_cache/debug/tick_history_{self.strategy_name}.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.today_ticks, f)
            print(f"当日tick数据已存储为 {pickle_file} 文件")
        else:
            json_file = f'./_cache/debug/tick_history_{self.strategy_name}.json'
            with open(json_file, 'w') as file:
                json.dump(self.today_ticks, file, indent=4)
            print(f"当日tick数据已存储为 {json_file} 文件")

    # -----------------------
    # 盘前下载数据缓存
    # -----------------------
    def _download_from_remote(
        self,
        target_codes: list,
        start: str,
        end: str,
        adjust: ExitRight,
        columns: list[str],
        data_source: DataSource,
    ):
        print(f'Prepared TIME RANGE: {start} - {end}')
        t0 = datetime.datetime.now()
        print(f'Downloading {len(target_codes)} stocks:')

        group_size = 200
        down_count = 0
        for i in range(0, len(target_codes), group_size):
            sub_codes = [sub_code for sub_code in target_codes[i:i + group_size]]
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
                time.sleep(0.5)
                if df is not None:
                    self.cache_history[code] = df
                    down_count += 1

        print(f'Download completed with {down_count} stock histories succeed!')
        t1 = datetime.datetime.now()
        print(f'Prepared TIME COST: {t1 - t0}')

    def download_cache_history(
        self,
        cache_path: str,  # DATA SOURCE 是tushare的时候不需要
        code_list: list[str],
        start: str,
        end: str,
        adjust: ExitRight,
        columns: list[str],
        data_source: DataSource,
    ):
        if data_source == DataSource.AKSHARE:
            temp_indicators = load_pickle(cache_path)
            if temp_indicators is not None and len(temp_indicators) > 0:
                # 如果有缓存就读缓存
                self.cache_history.clear()
                self.cache_history = {}
                self.cache_history.update(temp_indicators)
                print(f'{len(self.cache_history)} histories loaded from {cache_path}')
                if self.messager is not None:
                    self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                  f'历史{len(self.cache_history)}支')
            else:
                # 如果没缓存就刷新白名单
                self.cache_history.clear()
                self.cache_history = {}
                self._download_from_remote(code_list, start, end, adjust, columns, data_source)
                save_pickle(cache_path, self.cache_history)
                print(f'{len(self.cache_history)} of {len(code_list)} histories saved to {cache_path}')
                if self.messager is not None:
                    self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                  f'历史{len(self.cache_history)}支')
        elif data_source == DataSource.TUSHARE or data_source == DataSource.MOOTDX:
            hc = DailyHistoryCache()
            hc.set_data_source(data_source=data_source)
            if hc.daily_history is not None:
                hc.daily_history.download_recent_daily(20)  # 一个月数据
                # 下载后加载进内存
                start_date = datetime.datetime.strptime(start, '%Y%m%d')
                end_date = datetime.datetime.strptime(end, '%Y%m%d')
                delta = abs(end_date - start_date)
                self.cache_history = hc.daily_history.get_subset_copy(code_list, delta.days + 1)
        else:
            if self.messager is not None:
                self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                              f'无法识别数据源')

    @check_open_day
    def refresh_memory_history(
        self,
        code_list: list[str],
        start: str,
        end: str,
        data_source: DataSource,
    ):
        hc = DailyHistoryCache()
        hc.set_data_source(data_source=data_source)
        if hc.daily_history is not None:
            hc.daily_history.remove_recent_exit_right_histories(5)  # 一周数据
            # 重新加载进内存
            start_date = datetime.datetime.strptime(start, '%Y%m%d')
            end_date = datetime.datetime.strptime(end, '%Y%m%d')
            delta = abs(end_date - start_date)
            self.cache_history = hc.daily_history.get_subset_copy(code_list, delta.days + 1)


    # -----------------------
    # 盘后报告总结
    # -----------------------
    @check_open_day
    def daily_summary(self):
        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')

        try:
            if self.open_today_deal_report:
                self.daily_reporter.today_deal_report(today=curr_date)

            if self.delegate is not None:
                if self.open_today_hold_report:
                    positions = self.delegate.check_positions()
                    self.daily_reporter.today_hold_report(today=curr_date, positions=positions)

                asset = self.delegate.check_asset()
                self.daily_reporter.check_asset(today=curr_date, asset=asset)
            else:
                print('Missing delegate to complete reporting!')
        except Exception as e:
            print('Report failed: ', e)
            traceback.print_exc()

    # -----------------------
    # 定时器
    # -----------------------
    @check_open_day
    def before_trade_day_wrapper(self):
        if self.before_trade_day is not None:
            self.before_trade_day()

    @check_open_day
    def near_trade_begin_wrapper(self):
        if self.near_trade_begin is not None:
            self.near_trade_begin()

    @check_open_day
    def finish_trade_day_wrapper(self):
        if self.finish_trade_day is not None:
            self.finish_trade_day()

    def start_scheduler_without_qmt_data(self):
        run_time_ranges = [
            # 上午时间段: 09:15:00 到 11:29:59
            {
                'hour': '9',
                'minute': '15-59',
                'second': '0-59'  # 9:15到9:59的每秒
            },
            {
                'hour': '10',
                'minute': '0-59',
                'second': '0-59'  # 10:00到10:59的每秒
            },
            {
                'hour': '11',
                'minute': '0-29',
                'second': '0-59'  # 11:00到11:29的每秒（包含59秒）
            },
            # 下午时间段: 13:00:00 到 14:59:59
            {
                'hour': '13-14',
                'minute': '0-59',
                'second': '0-59'  # 13:00到14:59的每秒
            }
        ]

        for idx, cron_params in enumerate(run_time_ranges):
            self.scheduler.add_job(self.callback_run_no_quotes, 'cron', **cron_params, id=f"run_{idx}")

        if self.before_trade_day is not None:   # 03:00 ~ 06:59
            random_hour = random.randint(0, 3) + 3
            random_minute = random.randint(0, 59)
            self.scheduler.add_job(self.before_trade_day_wrapper, 'cron', hour=random_hour, minute=random_minute)

        if self.finish_trade_day is not None:   # 16:05 ~ 16:15
            random_minute = random.randint(0, 10) + 5
            self.scheduler.add_job(self.finish_trade_day_wrapper, 'cron', hour=16, minute=random_minute)

        self.scheduler.add_job(self.prev_check_open_day, 'cron', hour=1, minute=0, second=0)
        self.scheduler.add_job(self.callback_open_no_quotes, 'cron', hour=9, minute=14, second=59)
        self.scheduler.add_job(self.callback_close_no_quotes, 'cron', hour=11, minute=30, second=0)
        self.scheduler.add_job(self.callback_open_no_quotes, 'cron', hour=12, minute=59, second=59)
        self.scheduler.add_job(self.callback_close_no_quotes, 'cron', hour=15, minute=0, second=0)
        self.scheduler.add_job(self.daily_summary, 'cron', hour=15, minute=1, second=0)

        try:
            print('[定时器已启动]')
            self.scheduler.start()
        except KeyboardInterrupt:
            print('[手动结束进程]')
        except Exception as e:
            print('策略定时器出错：', e)
        finally:
            self.delegate.shutdown()

    def start_scheduler(self):
        if self.use_outside_data:
            self.start_scheduler_without_qmt_data()
            return

        # 默认定时任务列表
        cron_jobs = [
            ['01:00', self.prev_check_open_day, None],
            ['08:30', self.near_trade_begin_wrapper, None],
            ['09:15', self.subscribe_tick, None],
            ['11:30', self.unsubscribe_tick, (True, )],
            ['13:00', self.subscribe_tick, (True, )],
            ['15:00', self.unsubscribe_tick, None],
            ['15:01', self.daily_summary, None],
        ]
        if self.open_tick:
            cron_jobs.append(['09:10', self.clean_ticks_history, None])
            cron_jobs.append(['15:10', self.save_tick_history, None])

        if self.before_trade_day is not None:
            cron_jobs.append([  # 03:00 ~ 06:59
                f'0{random.randint(0, 3) + 3}:{random.randint(0, 59)}',
                self.before_trade_day_wrapper,
                None,
            ])  # random时间为了跑多个策略时防止短期预加载数据流量压力过大

        if self.finish_trade_day is not None:
            cron_jobs.append([  # 16:05 ~ 16:15
                f'16:{random.randint(0, 10) + 5}',
                self.finish_trade_day_wrapper,
                None,
            ])

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
                self.near_trade_begin_wrapper()
                if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
                    self.subscribe_tick()  # 重启时如果在交易时间则订阅Tick

            # 启动定时器
            try:
                print('[定时器已启动]')
                self.scheduler.start()
            except KeyboardInterrupt:
                print('[手动结束进程]')
            except Exception as e:
                print('策略定时器出错：', e)
            finally:
                self.delegate.shutdown()
                try:
                    import sys
                    sys.exit(0)
                except SystemExit:
                    import os
                    os._exit(0)
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
            #     print('[手动结束进程]')
            # finally:
            #     schedule.clear()
            #     self.delegate.shutdown()

    # -----------------------
    # 检查是否交易日
    # -----------------------
    def prev_check_open_day(self):
        now = datetime.datetime.now()
        curr_date = now.strftime('%Y-%m-%d')
        curr_time = now.strftime('%H:%M')
        print(f'[{curr_time}]', end='')
        is_open_day = check_is_open_day(curr_date)
        self.delegate.is_open_day = is_open_day


# -----------------------
# 持仓自动发现
# -----------------------
def update_position_held(lock: threading.Lock, delegate: XtDelegate, path: str):
    with lock:
        positions = delegate.check_positions()
        held_info = load_json(path)

        # 添加未被缓存记录的持仓：默认当日买入
        for position in positions:
            if position.can_use_volume > 0:
                if position.stock_code not in held_info.keys():
                    held_info[position.stock_code] = {InfoItem.DayCount: 0}

        # 删除已清仓的held_info记录
        if positions is not None and len(positions) > 0:
            position_codes = [position.stock_code for position in positions]
            print('当前持仓：', position_codes)
            holding_codes = list(held_info.keys())
            for code in holding_codes:
                if len(code) > 0 and code[0] != '_' and (code not in position_codes):
                    del held_info[code]
        else:
            print('当前空仓！')

        save_json(path, held_info)


# -----------------------
# 临时获取quotes
# -----------------------
def xt_get_ticks(code_list: list[str]) -> dict[str, any]:
    # http://docs.thinktrader.net/pages/36f5df/#%E8%8E%B7%E5%8F%96%E5%85%A8%E6%8E%A8%E6%95%B0%E6%8D%AE
    return xtdata.get_full_tick(code_list)
