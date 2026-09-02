import os
import time
import datetime
import random
import threading
from typing import Dict, Callable, List, Optional

import pandas as pd

from delegate.base_subscriber import HistorySubscriber
from delegate.am_nats_consumer import AmNatsConsumer
from delegate.daily_reporter import DailyReporter

from tools.utils_cache import check_is_open_day
from tools.utils_ding import BaseMessager
from tools.utils_remote import (
    is_tick_quote,
    qmt_quote_to_tick,
    QMT_TICK_DF_COLS,
    tick_quote_to_qmt_quote,
)

# [开始时间, 结束时间, 是否使用 AM 数据源]；False 使用 QMT xtdata
_DEFAULT_ARRANGEMENT = [
    ['09:14:30', '09:25:30', True],
    ['09:29:30', '11:30:30', False],
    ['12:59:30', '15:00:30', False],
]

SourceSlot = List  # [start_hms, end_hms, use_am: bool]


def _normalize_am_quotes(quotes: Dict) -> Dict:
    if not quotes:
        return quotes
    return {
        code: tick_quote_to_qmt_quote(quote) if is_tick_quote(quote) else quote
        for code, quote in quotes.items()
    }


def _parse_hms(hms: str) -> tuple[str, str, str]:
    parts = hms.split(':')
    if len(parts) == 2:
        return parts[0], parts[1], '0'
    return parts[0], parts[1], parts[2]


def _hms_to_time(hms: str) -> datetime.time:
    hr, mn, sec = _parse_hms(hms)
    return datetime.time(int(hr), int(mn), int(sec))


def _current_slot(arrangement: List[SourceSlot], now: datetime.datetime | None = None) -> bool | None:
    """返回当前时段是否应使用 AM；None 表示空档（不订阅）。"""
    t = (now or datetime.datetime.now()).time()
    for start, end, use_am in arrangement:
        if _hms_to_time(start) <= t < _hms_to_time(end):
            return use_am
    return None


class AmSubscriber(HistorySubscriber):
    def __init__(
        self,
        account_id: str,
        delegate=None,
        strategy_name: str = '',
        path_deal: str = '',
        path_assets: str = '',
        execute_strategy: Callable = None,
        execute_call_end: Callable = None,
        execute_interval: int = 1,
        before_trade_day: Callable = None,
        near_trade_begin: Callable = None,
        finish_trade_day: Callable = None,
        finish_call_hour: int = 15,
        use_ap_scheduler: bool = True,
        ding_messager: BaseMessager = None,
        open_middle_end_report: bool = False,
        open_today_deal_report: bool = False,
        open_today_hold_report: bool = False,
        today_report_show_bank: bool = False,
        open_tick_memory_cache: bool = False,
        tick_memory_data_frame: bool = False,
        nats_url: str = None,
        nats_subject: str = None,
        source_arrangement: List[SourceSlot] | None = None,
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
            finish_call_hour=finish_call_hour,
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
        self.quick_ticks: bool = False
        self.today_ticks: Dict[str, pd.DataFrame] = {}
        self._tick_rows: Dict[str, list[dict]] = {}
        self._tick_rows_materialized: Dict[str, int] = {}
        self.source_arrangement = source_arrangement or _DEFAULT_ARRANGEMENT

        self.lock_quotes_update = threading.Lock()
        self.cache_quotes: Dict[str, Dict] = {}
        self._last_quotes: Dict[str, Dict] = {}
        self._active_use_am: bool | None = None
        self._qmt_sub_sequence: int | None = None

        self.last_callback_time = datetime.datetime.now()

        self.__extend_codes = [
            '399001.SZ', '399006.SZ', '159101.SZ', '159315.SZ', '159915.SZ',
            '510500.SH', '510230.SH', '512680.SH', '588000.SH',
        ]
        self.code_list = ['000001.SH'] + self.__extend_codes

        consumer_kwargs = {'interval': float(execute_interval)}
        if nats_url is not None:
            consumer_kwargs['nats_url'] = nats_url
        if nats_subject is not None:
            consumer_kwargs['nats_subject'] = nats_subject
        self._consumer = AmNatsConsumer(**consumer_kwargs)

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

        self.tick_df_cols = list(QMT_TICK_DF_COLS)

    @property
    def active_source(self) -> str | None:
        if self._active_use_am is True:
            return 'AM'
        if self._active_use_am is False:
            return 'QMT'
        return None

    def _tick_rows_to_dataframe(self, rows: list) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=self.tick_df_cols)

    def materialize_today_ticks(self) -> None:
        """将尚未写入 today_ticks 的 list 行合并为 DataFrame（竞价结束 / 落盘前调用）。"""
        for code, rows in self._tick_rows.items():
            start = self._tick_rows_materialized.get(code, 0)
            if start >= len(rows):
                continue
            pending = rows[start:]
            new_df = self._tick_rows_to_dataframe(pending)
            existing = self.today_ticks.get(code)
            if isinstance(existing, pd.DataFrame) and len(existing) > 0:
                self.today_ticks[code] = pd.concat([existing, new_df], ignore_index=True)
            else:
                self.today_ticks[code] = new_df
            self._tick_rows_materialized[code] = len(rows)

    def get_tick_list(self, code: str) -> list[dict] | None:
        """盘中连续竞价因子用：返回当日完整 tick list（与 qmt_quote_to_tick 字段一致）。"""
        with self.lock_quotes_update:
            rows = self._tick_rows.get(code)
            if rows:
                return rows
            return None

    def get_tick_df(self, code: str) -> pd.DataFrame:
        """返回 today_ticks 中的 DataFrame；若尚未 materialize 则从 list 临时构建；无数据时返回空 DataFrame。"""
        empty = pd.DataFrame(columns=self.tick_df_cols)
        with self.lock_quotes_update:
            val = self.today_ticks.get(code)
            if isinstance(val, pd.DataFrame) and len(val) > 0:
                return val
            rows = self._tick_rows.get(code)
            if rows:
                return self._tick_rows_to_dataframe(rows)
        return empty

    def callback_sub_whole(self, quotes: Dict) -> None:
        now = datetime.datetime.now()
        self.last_callback_time = now

        curr_date = now.strftime('%Y-%m-%d')
        curr_time = now.strftime('%H:%M')

        if self.cache_limits['prev_minutes'] != curr_time:
            self.cache_limits['prev_minutes'] = curr_time
            print(f'\n[{curr_date} {curr_time}]', end='')

        curr_seconds = now.strftime('%S')
        if self._active_use_am is True:
            quotes = _normalize_am_quotes(quotes)
        with self.lock_quotes_update:
            self.cache_quotes.update(quotes)
            self._last_quotes.update(quotes)

        if self.open_tick and quotes and (not self.quick_ticks):
            self.record_tick_to_memory(quotes)

        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            print_mark = "'" if int(curr_seconds) % 10 == 9 else '.'
            print_mark = print_mark if len(self.cache_quotes) > 0 else 'x'

            if int(curr_seconds) % self.execute_interval == 0:
                is_clear = self.execute_strategy(curr_date, curr_time, curr_seconds, self.cache_quotes)

                if self.open_tick and self.quick_ticks and quotes:
                    self.record_tick_to_memory(quotes)

                # 开启 tick 缓存时全天保留 cache_quotes / _last_quotes，供策略读取最新价
                if is_clear and not self.open_tick:
                    with self.lock_quotes_update:
                        self.cache_quotes.clear()

                print(print_mark, end='')

    def callback_monitor(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        expected = _current_slot(self.source_arrangement)
        if expected is None:
            return

        if self._active_use_am != expected:
            self._switch_source(expected)
            return

        now = datetime.datetime.now()
        if (now - self.last_callback_time).total_seconds() <= 60:
            return

        return

    def _stop_active_source(self, pause: bool = False):
        if self._active_use_am is True:
            self._unsubscribe_am(pause=pause)
        elif self._active_use_am is False:
            self._unsubscribe_qmt(pause=pause)
        self._active_use_am = None

    def _switch_source(self, use_am: bool, resume: bool = False):
        if self._active_use_am == use_am:
            return
        self._stop_active_source(pause=resume)
        if use_am:
            self._subscribe_am(resume=resume)
        else:
            self._subscribe_qmt(resume=resume)
        self._active_use_am = use_am

    def _on_slot_start(self, use_am: bool):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        self._switch_source(use_am)

    def _on_slot_end(self, use_am: bool):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        if self._active_use_am == use_am:
            if self.open_tick:
                self.materialize_today_ticks()
            self._stop_active_source()

    def _apply_current_slot(self):
        expected = _current_slot(self.source_arrangement)
        if expected is None:
            self._stop_active_source()
        else:
            self._switch_source(expected)

    def _subscribe_am(self, resume: bool = False):
        label = 'AM'
        if self.messager is not None:
            self.messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name}:{"恢复" if resume else "开启"}{label} '
                f'{len(self.code_list)}支',
                output='[Message] BEGIN SUBSCRIBING\n')
        if self._consumer.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole) == 0:
            print(f'\n[开启{label}订阅] 订阅数:{len(self.code_list)}', end='')
        else:
            print(f'\n[开启{label}订阅] 失败，可能已有订阅', end='')

    def _unsubscribe_am(self, pause: bool = False):
        label = 'AM'
        if self._consumer.unsubscribe_quote() == 0:
            print(f'\n[结束{label}订阅] 订阅数:{len(self.code_list)}\n', end='')
            if self.messager is not None:
                self.messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name}:{"暂停" if pause else "关闭"}{label}',
                    output='[Message] END UNSUBSCRIBING\n')

    def _subscribe_qmt(self, resume: bool = False):
        from tools.utils_xtquant import xtdata

        label = 'QMT'
        if self.messager is not None:
            self.messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name}:{"恢复" if resume else "开启"}{label} '
                f'{len(self.code_list)}支',
                output='[Message] BEGIN SUBSCRIBING\n')
        xtdata.enable_hello = False
        self._qmt_sub_sequence = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        print(f'\n[开启{label}订阅] 订阅数:{len(self.code_list)} 订阅号:{self._qmt_sub_sequence}', end='')

    def _unsubscribe_qmt(self, pause: bool = False):
        from tools.utils_xtquant import xtdata

        label = 'QMT'
        if self._qmt_sub_sequence is not None:
            xtdata.unsubscribe_quote(self._qmt_sub_sequence)
            self._qmt_sub_sequence = None
            print(f'\n[结束{label}订阅] 订阅数:{len(self.code_list)}\n', end='')
            if self.messager is not None:
                self.messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name}:{"暂停" if pause else "关闭"}{label}',
                    output='[Message] END UNSUBSCRIBING\n')

    def subscribe_tick(self, resume: bool = False):
        self._apply_current_slot()

    def unsubscribe_tick(self, pause: bool = False):
        self._stop_active_source(pause=pause)

    def resubscribe_tick(self, notice: bool = True):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        if self._active_use_am is True:
            self._consumer.unsubscribe_quote()
            self._subscribe_am(resume=notice)
        elif self._active_use_am is False:
            self._unsubscribe_qmt()
            self._subscribe_qmt(resume=notice)
        if self.messager is not None and notice and self.active_source:
            self.messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name}:重启{self.active_source} {len(self.code_list)}支',
                output='\n[Message] FINISH RESUBSCRIBING')
            print(f'\n[重启行情订阅] 数据源:{self.active_source} 订阅数:{len(self.code_list)}', end='')

    def update_code_list(self, code_list: list[str]):
        print(f'[订阅更新] {code_list}\n', end='')
        self.code_list = ['000001.SH'] + code_list
        extend = 10 - len(self.code_list)
        if extend > 0:
            self.code_list.extend(self.__extend_codes[:extend])
        self._consumer.update_code_list(self.code_list)

    def _tick_unchanged_since_last(self, code: str, quote: dict) -> bool:
        rows = self._tick_rows.get(code)
        if not rows:
            return False
        tick = qmt_quote_to_tick(quote)
        last = rows[-1]
        return all(last.get(col) == tick.get(col) for col in self.tick_df_cols if col != 'local')

    def record_tick_to_memory(self, quotes):
        if not quotes:
            return

        local_time = datetime.datetime.now().strftime('%H:%M:%S')
        for code, quote in quotes.items():
            if self._tick_unchanged_since_last(code, quote):
                continue
            tick = qmt_quote_to_tick(quote)
            tick['local'] = local_time
            self._tick_rows.setdefault(code, []).append(tick)

    def clean_ticks_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        self.today_ticks.clear()
        self.today_ticks = {}
        self._tick_rows.clear()
        self._tick_rows_materialized.clear()
        self._last_quotes.clear()
        print('[提示] 已清除tick缓存')

    def save_tick_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        self.materialize_today_ticks()

        out_dir = './_cache/debug'
        os.makedirs(out_dir, exist_ok=True)
        parquet_file = f'{out_dir}/ticks_{self.strategy_name}_{datetime.datetime.now().strftime("%A").lower()}.parquet'

        with self.lock_quotes_update:
            ticks_snapshot = {
                code: df.copy()
                for code, df in (self.today_ticks or {}).items()
                if isinstance(df, pd.DataFrame) and len(df) > 0
            }

        frames: list[pd.DataFrame] = []
        for code, df in ticks_snapshot.items():
            code_df = df.copy()
            code_df.insert(0, 'code', code)
            frames.append(code_df)

        if not frames:
            print(f'[提示] 当日tick数据为空，未写入 {parquet_file}')
            return

        all_df = pd.concat(frames, ignore_index=True)
        try:
            all_df.to_parquet(parquet_file, index=False, compression='zstd', engine='pyarrow')
        except Exception as e:
            print(f'[提示] 当日tick数据zstd压缩存储失败，改用snappy压缩存储：', e)
            all_df.to_parquet(parquet_file, index=False, compression='snappy', engine='pyarrow')
        print(f'[提示] 当日tick数据已存储为 {parquet_file} 文件 rows={len(all_df)}')

    def clear_all(self):
        super().clear_all()
        self.cache_quotes.clear()
        self._last_quotes.clear()
        self._tick_rows.clear()
        self._tick_rows_materialized.clear()
        self.cache_history.clear()
        self.today_ticks.clear()
        self.code_list = ['000001.SH'] + self.__extend_codes

    def execute_call_end_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        self.materialize_today_ticks()
        if self.execute_call_end is not None:
            print('[定时任务] 竞价任务开始')
            self.execute_call_end()
            try:
                print('[定时任务] 竞价任务完成\n', end='')
            except Exception as e:
                print(f'[定时任务] 竞价任务出错: {e}\n', end='')

    def _add_arrangement_jobs(self, cron_jobs: list):
        for start, end, use_am in self.source_arrangement:
            cron_jobs.append([start, self._on_slot_start, (use_am,)])
            cron_jobs.append([end, self._on_slot_end, (use_am,)])

    def _start_scheduler(self):
        cron_jobs = [
            ['01:00', self.prev_check_open_day, None],
            ['08:30', self.near_trade_begin_wrapper, None],
            ['15:02', self.daily_summary, None],
        ]

        self._add_arrangement_jobs(cron_jobs)

        if self.open_tick:
            cron_jobs.append(['09:10', self.clean_ticks_history, None])
            cron_jobs.append(['15:10', self.save_tick_history, None])

        if self.before_trade_day is not None:
            before_time = f'0{random.randint(0, 3) + 3}:{random.randint(0, 59)}'
            cron_jobs.append([before_time, self.before_trade_day_wrapper, None])

        if self.check_before_finished is not None:
            check_before_time = f'08:{random.randint(0, 59)}'
            cron_jobs.append([check_before_time, self.check_before_finished, None])

        if self.finish_trade_day is not None:
            finish_time = f'{self.finish_call_hour}:{random.randint(0, 10) + 15}'
            cron_jobs.append([finish_time, self.finish_trade_day_wrapper, None])

        if self.open_middle_end_report:
            cron_jobs.append(['11:32', self.daily_summary, None])

        for cron_job in cron_jobs:
            hr, mn, sec = _parse_hms(cron_job[0])
            kwargs = {'hour': hr, 'minute': mn, 'second': sec}
            if cron_job[2] is None:
                self.scheduler.add_job(cron_job[1], 'cron', **kwargs)
            else:
                self.scheduler.add_job(cron_job[1], 'cron', **kwargs, args=list(cron_job[2]))

        if self.execute_call_end is not None:
            self.scheduler.add_job(self.execute_call_end_wrapper, 'cron', hour=9, minute=25, second=45)

        monitor_time_list = [
            '09:35', '09:45', '09:55', '10:05', '10:15', '10:25',
            '10:35', '10:45', '10:55', '11:05', '11:15', '11:25',
            '13:05', '13:15', '13:25', '13:35', '13:45', '13:55',
            '14:05', '14:15', '14:25', '14:35', '14:45', '14:55',
        ]
        for monitor_time in monitor_time_list:
            hr, mn, sec = _parse_hms(monitor_time)
            self.scheduler.add_job(self.callback_monitor, 'cron', hour=hr, minute=mn, second=sec)

        try:
            self._print_scheduled_jobs()
            print('[定时任务] 计划启动')
            self.scheduler.start()
        except KeyboardInterrupt:
            print('[定时任务] 手动结束')
            os.system('pause')
        except Exception as e:
            print('[定时任务] 执行出错：', e)
            os.system('pause')
        finally:
            if self.delegate is not None:
                self.delegate.shutdown()
            print('[定时任务] 关闭完成')
            try:
                import sys
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def start_scheduler(self):
        temp_now = datetime.datetime.now()
        temp_date = temp_now.strftime('%Y-%m-%d')
        temp_time = temp_now.strftime('%H:%M')
        if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
            self.prev_check_open_day()
            self.before_trade_day_wrapper()
            self.near_trade_begin_wrapper()
            if _current_slot(self.source_arrangement, temp_now) is not None:
                self._apply_current_slot()
        self._start_scheduler()
