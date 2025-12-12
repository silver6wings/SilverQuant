import time
import datetime
import json
import pickle
import random
import threading
from typing import Dict, Callable, Optional

import pandas as pd
from xtquant import xtdata

from delegate.base_subscriber import HistorySubscriber
from delegate.xt_delegate import XtDelegate
from delegate.daily_reporter import DailyReporter

from tools.utils_cache import check_is_open_day
from tools.utils_ding import BaseMessager
from tools.utils_remote import qmt_quote_to_tick


class XtSubscriber(HistorySubscriber):
    def __init__(
        self,
        # 基本信息
        account_id: str,
        delegate: Optional[XtDelegate],
        strategy_name: str,
        path_deal: str,
        path_assets: str,
        # 回调
        execute_strategy: Callable,         # 策略回调函数
        execute_call_end: Callable = None,  # 策略竞价结束回调
        execute_interval: int = 1,          # 策略执行间隔，单位（秒）
        before_trade_day: Callable = None,  # 盘前函数
        near_trade_begin: Callable = None,  # 盘后函数
        finish_trade_day: Callable = None,  # 盘后函数
        # 订阅
        use_ap_scheduler: bool = False,     # 默认使用旧版 schedule （尽可能向前兼容旧策略吧）
        # 通知
        ding_messager: BaseMessager = None,
        # 日报
        open_middle_end_report: bool = False,   # 午盘结束的报告
        open_today_deal_report: bool = False,   # 每日交易记录报告
        open_today_hold_report: bool = False,   # 每日持仓记录报告
        today_report_show_bank: bool = False,   # 是否显示银行流水（国金QMT会卡死所以默认关闭）
        # tick 缓存
        open_tick_memory_cache: bool = False,
        tick_memory_data_frame: bool = False,
    ):
        super().__init__(
            account_id=account_id,
            delegate=delegate,
            strategy_name=strategy_name,
            path_deal=path_deal,
            path_assets=path_assets,
            execute_strategy=execute_strategy,
            execute_call_end=execute_call_end,
            execute_interval=execute_interval,
            before_trade_day=before_trade_day,
            near_trade_begin=near_trade_begin,
            finish_trade_day=finish_trade_day,
            open_middle_end_report=open_middle_end_report,
            open_today_deal_report=open_today_deal_report,
            open_today_hold_report=open_today_hold_report,
            today_report_show_bank=today_report_show_bank,
            ding_messager=ding_messager,
        )
        self.use_ap_scheduler = use_ap_scheduler
        self.create_scheduler()

        self.open_tick = open_tick_memory_cache
        self.is_ticks_df = tick_memory_data_frame
        self.quick_ticks: bool = False                          # 是否开启quick tick模式
        self.today_ticks: Dict[str, list | pd.DataFrame] = {}   # 记录tick的历史信息
        self.lock_quotes_update = threading.Lock()  # 聚合实时打点缓存的锁

        self.cache_quotes: Dict[str, Dict] = {}     # 记录实时的价格信息

        self.code_list = ['000001.SH']  # 默认只有上证指数
        self.last_callback_time = datetime.datetime.now()       # 上次返回quotes 时间

        self.__extend_codes = ['399001.SZ', '510230.SH', '512680.SH', '159915.SZ', '510500.SH',
                               '588000.SH', '159101.SZ', '399006.SZ', '159315.SZ']

        self.daily_reporter = DailyReporter(
            account_id=self.account_id,
            delegate=self.delegate,
            strategy_name=self.strategy_name,
            path_deal=self.path_deal,
            path_assets=self.path_assets,
            messager=self.messager,
            use_outside_data=False,
            today_report_show_bank=self.today_report_show_bank,
        )

        if self.is_ticks_df:
            self.tick_df_cols = ['time', 'price', 'high', 'low', 'volume', 'amount'] \
                + [f'askPrice{i}' for i in range(1, 6)] \
                + [f'askVol{i}' for i in range(1, 6)] \
                + [f'bidPrice{i}' for i in range(1, 6)] \
                + [f'bidVol{i}' for i in range(1, 6)]

        self.curr_trade_date = '1990-12-19' #记录当前股票交易日期

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
                    self.record_tick_to_memory(self.cache_quotes)

                # str(%Y-%m-%d) str(%H:%M) str(%S) dict(code: quotes)
                is_clear = self.execute_strategy(curr_date, curr_time, curr_seconds, self.cache_quotes)

                # 更快（先执行再记录）
                if self.open_tick and self.quick_ticks:
                    self.record_tick_to_memory(self.cache_quotes)

                if is_clear:
                    with self.lock_quotes_update:
                        self.cache_quotes.clear()  # execute_strategy() return True means need clear

                print(print_mark, end='')  # 每秒钟开始的时候输出一个点

    # -----------------------
    # 监测主策略执行
    # -----------------------
    def callback_monitor(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        now = datetime.datetime.now()
        callback_timedelta = (now - self.last_callback_time).total_seconds()
        if callback_timedelta > 60:
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
    # 订阅 tick 相关
    # -----------------------
    def subscribe_tick(self, resume: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                          f'{"恢复" if resume else "开启"} {len(self.code_list)}支')
        print('[开启行情订阅]', end='')
        xtdata.enable_hello = False
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)

    def unsubscribe_tick(self, pause: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if 'sub_seq' in self.cache_limits:
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
            print('\n[结束行情订阅]')
            if self.messager is not None:
                self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                              f'{"暂停" if pause else "关闭"}')

    def resubscribe_tick(self, notice: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

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
    # 盘中实时的 tick 历史
    # -----------------------
    def record_tick_to_memory(self, quotes):
        # 记录 tick 历史到内存
        if self.is_ticks_df:
            for code in quotes:
                quote = quotes[code]
                tick = qmt_quote_to_tick(quote)
                new_tick_df = pd.DataFrame([tick], columns=self.tick_df_cols)
                if code not in self.today_ticks:
                    self.today_ticks[code] = new_tick_df
                else:
                    self.today_ticks[code] = pd.concat([self.today_ticks[code], new_tick_df], ignore_index=True)
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

    def clean_ticks_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        self.today_ticks.clear()
        self.today_ticks = {}
        print(f"[提示] 已清除tick缓存")

    def save_tick_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.is_ticks_df:
            pickle_file = f'./_cache/debug/tick_history_{self.strategy_name}.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.today_ticks, f)
            print(f"[提示] 当日tick数据已存储为 {pickle_file} 文件")
        else:
            json_file = f'./_cache/debug/tick_history_{self.strategy_name}.json'
            with open(json_file, 'w') as file:
                json.dump(self.today_ticks, file, indent=4)
            print(f"[提示] 当日tick数据已存储为 {json_file} 文件")

    # 检查是否完成盘前准备
    def check_before_finished(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if (self.before_trade_day is not None or self.near_trade_begin is not None) \
            and (
                self.curr_trade_date != datetime.datetime.now().strftime("%Y-%m-%d")
                or len(self.cache_history) < 1
            ):
            print('[警告] 盘前准备未完成，尝试重新执行盘前函数')
            self.before_trade_day_wrapper()
            self.near_trade_begin_wrapper()
        print(f'[提示] 当前交易日：{self.curr_trade_date}')

    # -----------------------
    # 定时器
    # -----------------------
    def before_trade_day_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        self.cache_quotes.clear()
        self.cache_history.clear()
        self.today_ticks.clear()
        self.history_day_klines.clear()
        self.code_list = ['000001.SH']  # 默认只有上证指数

        if self.before_trade_day is not None:
            self.before_trade_day()
            self.curr_trade_date = datetime.datetime.now().strftime('%Y-%m-%d')


    def _start_qmt_scheduler(self):
        # 默认定时任务列表
        cron_jobs = [
            ['01:00', self.prev_check_open_day, None],
            ['08:30', self.near_trade_begin_wrapper, None],
            ['08:55', self.check_before_finished, None],
            ['09:14', self.subscribe_tick, None],
            ['11:31', self.unsubscribe_tick, (True, )],
            ['12:59', self.subscribe_tick, (True, )],
            ['15:01', self.unsubscribe_tick, None],
            ['15:02', self.daily_summary, None],
        ]

        if self.open_tick:
            cron_jobs.append(['09:10', self.clean_ticks_history, None])
            cron_jobs.append(['15:10', self.save_tick_history, None])

        if self.before_trade_day is not None:
            # random 时间为了跑多个策略时防止短期预加载数据流量压力过大
            before_time = f'0{random.randint(0, 3) + 3}:{random.randint(0, 59)}'  # 03:00 ~ 06:59
            cron_jobs.append([before_time, self.before_trade_day_wrapper, None])

        if self.finish_trade_day is not None:
            # random 时间为了跑多个策略时防止短期预加载数据流量压力过大
            finish_time = f'16:{random.randint(0, 10) + 5}'  # 16:05 ~ 16:15
            cron_jobs.append([finish_time, self.finish_trade_day_wrapper, None])

        if self.execute_call_end is not None:
            cron_jobs.append(['09:26', self.execute_call_end_wrapper, None])

        if self.open_middle_end_report:
            cron_jobs.append(['11:32', self.daily_summary, None])

        # 数据源中断检查时间点
        monitor_time_list = [
            '09:35', '09:45', '09:55', '10:05', '10:15', '10:25',
            '10:35', '10:45', '10:55', '11:05', '11:15', '11:25',
            '13:05', '13:15', '13:25', '13:35', '13:45', '13:55',
            '14:05', '14:15', '14:25', '14:35', '14:45', '14:55',
        ]

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

    def start_scheduler(self):
        if self.use_ap_scheduler:
            temp_now = datetime.datetime.now()
            temp_date = temp_now.strftime('%Y-%m-%d')
            temp_time = temp_now.strftime('%H:%M')
            # 盘中执行需要补齐
            if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
                self.before_trade_day_wrapper()
                self.near_trade_begin_wrapper()
                if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
                    self.subscribe_tick()  # 重启时如果在交易时间则订阅Tick
        self._start_qmt_scheduler()


# -----------------------
# 临时获取quotes
# -----------------------
def xt_get_ticks(code_list: list[str]) -> dict[str, any]:
    # http://docs.thinktrader.net/pages/36f5df/#%E8%8E%B7%E5%8F%96%E5%85%A8%E6%8E%A8%E6%95%B0%E6%8D%AE
    return xtdata.get_full_tick(code_list)
