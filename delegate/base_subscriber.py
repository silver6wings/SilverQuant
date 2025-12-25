import time
import datetime
import random
import threading
import traceback
from typing import Dict, Callable, Optional

import pandas as pd

from delegate.base_delegate import BaseDelegate
from delegate.daily_reporter import DailyReporter
from delegate.daily_history import DailyHistoryCache

from tools.utils_cache import StockNames, check_is_open_day, load_pickle, save_pickle, get_trading_date_list
from tools.utils_ding import BaseMessager
from tools.utils_mootdx import get_tdxzip_history
from tools.utils_remote import DataSource, ExitRight, get_daily_history


class BaseSubscriber:
    def __init__(
        self,
        # 基本信息
        account_id: str,
        delegate: Optional[BaseDelegate],
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
        # 非QMT
        custom_sub_begin: Callable = None,  # 使用外部数据时的自定义启动
        custom_unsub_end: Callable = None,  # 使用外部数据时的自定义关闭
        # 通知
        ding_messager: BaseMessager = None,
        # 日报
        open_middle_end_report: bool = False,   # 午盘结束的报告
        open_today_deal_report: bool = False,   # 每日交易记录报告
        open_today_hold_report: bool = False,   # 每日持仓记录报告
        today_report_show_bank: bool = False,   # 是否显示银行流水（国金QMT会卡死所以默认关闭）
    ):
        self.account_id = '**' + str(account_id)[-4:]
        self.delegate = delegate
        if self.delegate is not None:
            self.delegate.subscriber = self

        self.strategy_name = strategy_name
        self.path_deal = path_deal
        self.path_assets = path_assets

        self.execute_strategy = execute_strategy
        self.execute_call_end = execute_call_end
        self.execute_interval = execute_interval
        self.before_trade_day = before_trade_day    # 提前准备某些耗时长的任务
        self.near_trade_begin = near_trade_begin    # 有些数据临近开盘才更新，这里保证内存里的数据正确
        self.finish_trade_day = finish_trade_day    # 盘后及时做一些总结汇报入库类的整理工作

        self.custom_begin_sub = custom_sub_begin
        self.custom_end_unsub = custom_unsub_end

        self.open_middle_end_report = open_middle_end_report
        self.open_today_deal_report = open_today_deal_report
        self.open_today_hold_report = open_today_hold_report
        self.today_report_show_bank = today_report_show_bank

        self.messager = ding_messager

        self.scheduler = None
        self.create_scheduler()

        self.stock_names = StockNames()
        self.daily_reporter = DailyReporter(
            account_id=self.account_id,
            delegate=self.delegate,
            strategy_name=self.strategy_name,
            path_deal=self.path_deal,
            path_assets=self.path_assets,
            messager=self.messager,
            use_outside_data=True,
            today_report_show_bank=self.today_report_show_bank,
        )

        self.cache_limits: Dict[str, str] = {       # 限制执行次数的缓存集合
            'prev_seconds': '',                     # 限制每秒一次跑策略扫描的缓存
            'prev_minutes': '',                     # 限制每分钟屏幕心跳换行的缓存
        }
        self.cache_history: Dict[str, pd.DataFrame] = {}    # 记录历史日线行情的信息 { code: DataFrame }

        # 这个成员变量区别于cache_history，保存全部股票的日线数据550天，cache_history只包含code_list中指定天数数据
        self.history_day_klines : Dict[str, pd.DataFrame] = {}

        self.code_list = []
        self.curr_trade_date = ''

    # -----------------------
    # 监测主策略执行
    # -----------------------
    def callback_run_no_quotes(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        now = datetime.datetime.now()

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
                print('.', end='')  # cache_quotes 肯定没数据，这里就是输出观察线程健康
                # str(%Y-%m-%d), str(%H:%M), str(%S)
                self.execute_strategy(curr_date, curr_time, curr_seconds, {})

    def callback_open_no_quotes(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:开启', output='[MSG START]\n')

        print('[启动策略]')

        if self.custom_begin_sub is not None:
            threading.Thread(target=self.custom_begin_sub).start()

    def callback_close_no_quotes(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        print('\n[关闭策略]')

        if self.messager is not None:
            self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:结束', output='[MSG STOP]\n')

        if self.custom_end_unsub is not None:
            threading.Thread(target=self.custom_end_unsub).start()

    def update_code_list(self, code_list: list[str]):
        self.code_list = code_list

    # -----------------------
    # 任务接口
    # -----------------------
    def before_trade_day_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.before_trade_day is not None:
            self.before_trade_day()
            self.curr_trade_date = datetime.datetime.now().strftime('%Y-%m-%d')

    def near_trade_begin_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.near_trade_begin is not None:
            self.near_trade_begin()
            if self.before_trade_day is None:  # 没有设置before_trade_day 情况
                self.curr_trade_date = datetime.datetime.now().strftime('%Y-%m-%d')
            print(f'今日盘前准备工作已完成')

    def finish_trade_day_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.finish_trade_day is not None:
            self.finish_trade_day()

    def execute_call_end_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        print('[竞价结束回调]')
        if self.execute_call_end is not None:
            self.execute_call_end()

    # 检查是否完成盘前准备
    def check_before_finished(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if (self.before_trade_day is not None or self.near_trade_begin is not None) and \
                (self.curr_trade_date != datetime.datetime.now().strftime("%Y-%m-%d")):
            print('[ERROR]盘前准备未完成，尝试重新执行盘前函数')
            self.before_trade_day_wrapper()
            self.near_trade_begin_wrapper()
        print(f'当前交易日：[{self.curr_trade_date}]')

    # -----------------------
    # 盘后报告总结
    # -----------------------
    def daily_summary(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')

        if self.open_today_deal_report:
            try:
                self.daily_reporter.today_deal_report(today=curr_date)
            except Exception as e:
                print('Report deal failed: ', e)
                traceback.print_exc()

        if self.open_today_hold_report:
            try:
                if self.delegate is not None:
                    positions = self.delegate.check_positions()
                    self.daily_reporter.today_hold_report(today=curr_date, positions=positions)
                else:
                    print('Missing delegate to complete reporting!')
            except Exception as e:
                print('Report position failed: ', e)
                traceback.print_exc()

        try:
            if self.delegate is not None:
                asset = self.delegate.check_asset()
                self.daily_reporter.check_asset(today=curr_date, asset=asset)
        except Exception as e:
            print('Report asset failed: ', e)
            traceback.print_exc()

    # -----------------------
    # 定时器
    # -----------------------
    def _start_scheduler(self):
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

        if self.execute_call_end is not None:
            self.scheduler.add_job(self.execute_call_end_wrapper(), 'cron', hour=9, minute=25, second=30)

        if self.open_middle_end_report:
            self.scheduler.add_job(self.daily_summary, 'cron', hour=11, minute=32)

        self.scheduler.add_job(self.prev_check_open_day, 'cron', hour=1, minute=0)
        self.scheduler.add_job(self.check_before_finished, 'cron', hour=8, minute=55) # 检查当天是否完成准备
        self.scheduler.add_job(self.callback_open_no_quotes, 'cron', hour=9, minute=14, second=30)
        self.scheduler.add_job(self.callback_close_no_quotes, 'cron', hour=11, minute=30, second=30)
        self.scheduler.add_job(self.callback_open_no_quotes, 'cron', hour=12, minute=59, second=30)
        self.scheduler.add_job(self.callback_close_no_quotes, 'cron', hour=15, minute=0, second=30)
        self.scheduler.add_job(self.daily_summary, 'cron', hour=15, minute=2)

        try:
            print('[定时器已启动]')
            self.scheduler.start()
        except KeyboardInterrupt:
            print('[手动结束进程]')
        except Exception as e:
            print('策略定时器出错：', e)
        finally:
            self.delegate.shutdown()

    def create_scheduler(self):
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.executors.pool import ThreadPoolExecutor
        executors = {
            'default': ThreadPoolExecutor(32),
        }
        job_defaults = {
            'coalesce': True,
            'misfire_grace_time': 180,
            'max_instances': 3
        }
        self.scheduler = BlockingScheduler(timezone='Asia/Shanghai', executors=executors, job_defaults=job_defaults)

    def start_scheduler(self):
        temp_now = datetime.datetime.now()
        temp_date = temp_now.strftime('%Y-%m-%d')
        temp_time = temp_now.strftime('%H:%M')
        # 盘中执行需要补齐
        if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
            self.before_trade_day_wrapper()
            self.near_trade_begin_wrapper()
            if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
                self.callback_open_no_quotes()  # 重启时如果在交易时间则订阅Tick

        self._start_scheduler()

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


class HistorySubscriber(BaseSubscriber):
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

    def _download_from_tdx(self, target_codes: list, start: str, end: str, adjust: str, columns: list[str]):
        print(f'Prepared time range: {start} - {end}')
        t0 = datetime.datetime.now()

        full_history = get_tdxzip_history(adjust=adjust)
        self.history_day_klines = full_history

        days = len(get_trading_date_list(start, end))

        i = 0
        for code in target_codes:
            if code in full_history:
                i += 1
                self.cache_history[code] = full_history[code][columns].tail(days).copy()
        print(f'[HISTORY] Find {i}/{len(target_codes)} codes returned.')

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
        # ======== 每日一次性全量数据源 ========
        if data_source == DataSource.AKSHARE or data_source == DataSource.TDXZIP:
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
                if data_source == DataSource.AKSHARE:
                    self._download_from_remote(code_list, start, end, adjust, columns, data_source)
                else:
                    print('[提示] 使用TDX ZIP文件作为数据源，请在RUN代码中添加调度任务 check_xdxr_cache '
                          '更新除权除息数据，建议运行时段在05:30之后')
                    print('[提示] 使用TDX ZIP文件作为数据源，请在RUN代码中建议在 near_trade_begin 中执行 '
                          'download_cache_history 获取历史数据，避免 before_trade_day 执行时间太早未更新除权信息')
                    self._download_from_tdx(code_list, start, end, adjust, columns)

                save_pickle(cache_path, self.cache_history)
                print(f'{len(self.cache_history)} of {len(code_list)} histories saved to {cache_path}')
                if self.messager is not None:
                    self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:'
                                                  f'历史{len(self.cache_history)}支')

            if data_source == DataSource.TDXZIP and self.history_day_klines is None:
                self.history_day_klines = get_tdxzip_history(adjust=adjust)

        # ======== 预加载每日增量数据源 ========
        elif data_source == DataSource.TUSHARE or data_source == DataSource.MOOTDX:
            hc = DailyHistoryCache()
            hc.set_data_source(data_source=data_source)
            if hc.daily_history is not None:
                hc.daily_history.remove_recent_exit_right_histories(5)  # 一周数据
                hc.daily_history.download_recent_daily(20)  # 一个月数据
                # 下载后加载进内存
                start_date = datetime.datetime.strptime(start, '%Y%m%d')
                end_date = datetime.datetime.strptime(end, '%Y%m%d')
                delta = abs(end_date - start_date)
                self.cache_history = hc.daily_history.get_subset_copy(code_list, delta.days + 1)
        else:
            if self.messager is not None:
                self.messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:\n无法识别的数据源')


    # 重新加载历史数据进内存
    def refresh_memory_history(self, code_list: list[str], start: str, end: str, data_source: DataSource):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        hc = DailyHistoryCache()
        hc.set_data_source(data_source=data_source)
        if hc.daily_history is not None:
            start_date = datetime.datetime.strptime(start, '%Y%m%d')
            end_date = datetime.datetime.strptime(end, '%Y%m%d')
            delta = abs(end_date - start_date)
            self.cache_history = hc.daily_history.get_subset_copy(code_list, delta.days + 1)
