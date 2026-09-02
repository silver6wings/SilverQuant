"""
Tushare 专用历史日线缓存：继承 DailyHistory，增量/单日采用小批量 API + 逐股 fallback。

- 磁盘 CSV 格式与其他数据源一致（列名、路径相同）
- 价格为不复权（BFQ）；Tushare daily 接口不提供前复权
- 增量 18:59 后包含当日 bar（仅 TS）；其他数据源截止到上一交易日
- 全量 download_all_to_disk：10 只/批 + 单股 fallback，默认 interval 1.2s
- 增量 download_recent_daily：flat 批量（无二分拆分）+ 逐股 fallback
"""
import datetime
import os
import time
import traceback
from typing import Optional

import pandas as pd

from delegate.daily_history import DailyHistory
from tools.utils_cache import get_prev_trading_date
from tools.utils_remote import DataSource, get_ts_daily_histories
from tools.utils_remote_ts import get_ts_daily_histories_flat


class DailyHistoryTS(DailyHistory):
    """Tushare 优化版 DailyHistory：全量/增量/单日均有独立实现。"""

    default_data_source = DataSource.TUSHARE
    _log_prefix = '[历史日线TS]'
    ts_batch_group_size = 10
    ts_download_interval: float = 1.2

    def __init__(
        self,
        root_path: str = DailyHistory.default_root_path,
        init_day_count: int = 550,
    ):
        super().__init__(
            root_path=root_path,
            data_source=DataSource.TUSHARE,
            init_day_count=init_day_count,
        )
        self._disk_last_datetime: dict[str, int] = {}

    def download_all_to_disk(self, renew_code_list: bool = True, interval: float | None = None) -> None:
        if interval is None:
            interval = self.ts_download_interval
        super().download_all_to_disk(renew_code_list=renew_code_list, interval=interval)

    def _download_codes(self, code_list: list[str], day_count: int, interval: float | None = None) -> None:
        """Tushare 全量：10 只/批，批内缺失再单股 fallback。"""
        if interval is None:
            interval = self.ts_download_interval
        now = datetime.datetime.now()
        forward_day = 1
        start_date = get_prev_trading_date(now, forward_day + day_count)
        end_date = get_prev_trading_date(now, forward_day)

        downloaded_count = 0
        download_failure: list[str] = []
        group_size = self.ts_batch_group_size

        for i in range(0, len(code_list), group_size):
            group_codes = code_list[i:i + group_size]
            dfs = get_ts_daily_histories(
                group_codes,
                start_date,
                end_date,
                columns=self.default_columns,
                interval=interval,
            )
            for code in group_codes:
                df = dfs.get(code)
                if df is None or len(df) == 0:
                    df = self._download_single_code_daily(code, start_date, end_date, interval)
                if df is None or len(df) == 0:
                    download_failure.append(code)
                    continue
                self._save_downloaded_daily(code, df)
                downloaded_count += 1

            print(
                f'{self._log_prefix} [{downloaded_count}/{min(i + group_size, len(code_list))}]',
                group_codes,
            )

        print(
            f'{self._log_prefix} Download finished with {len(download_failure)} fails: '
            f'{download_failure}',
        )

    @staticmethod
    def _incremental_end_forward(now: datetime.datetime) -> int:
        """18:59 之后包含当日数据（仅 Tushare 增量）。"""
        return 0 if now.hour > 18 else 1

    def load_history_from_disk_to_memory(self, auto_update: bool = True) -> None:
        if auto_update:
            print(
                f'{self._log_prefix} 已跳过 auto_update 逐股补下载'
                '（缺文件请用 download_all_to_disk / download_recent_daily）',
            )
        super().load_history_from_disk_to_memory(auto_update=False)

    def _download_local_missed(self) -> None:
        code_list = self.get_code_list()
        on_disk = self._codes_on_disk()
        missing = [code for code in code_list if code not in on_disk]
        if not missing:
            return
        print(
            f'{self._log_prefix} 本地缺 {len(missing)} 只 csv，'
            f'请运行 download_all_to_disk 或 download_recent_daily 补齐',
        )

    def _fetch_single_code(
        self,
        code: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        df = self._download_single_code_daily(
            code,
            start_date,
            end_date,
            int(self.ts_download_interval),
        )
        if df is None or len(df) == 0:
            return None
        if not self._has_datetime_column(df, code, 'ts fetch'):
            return None
        return df

    def _prepare_code_df_for_save(
        self,
        code: str,
        code_df: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """校验末交易日；缺失则单股重试；仍缺则按停牌规则放行或拒绝。"""
        if self._has_end_date_bar(code_df, end_date):
            return code_df, None

        last_int = self._last_datetime(code_df)
        retried = self._fetch_single_code(code, start_date, end_date)
        if retried is not None and len(retried) > 0:
            code_df = retried

        if self._has_end_date_bar(code_df, end_date):
            return code_df, None

        if code_df is not None and len(code_df) > 0 and self._accept_suspend_tail(code_df, end_date):
            new_last = self._last_datetime(code_df)
            print(
                f'{code} 缺 {end_date}（末条 {new_last}），'
                f'视为停牌/无成交，允许保存',
                end='',
                flush=True,
            )
            return code_df, None

        if code_df is None or len(code_df) == 0:
            return None, 'empty'
        return None, f'missing_tail(last={last_int}, need={end_date})'

    def _recent_trading_date_ints(self, now: datetime.datetime, lookback: int) -> list[int]:
        end_forward = self._incremental_end_forward(now)
        dates = [
            int(get_prev_trading_date(now, forward))
            for forward in range(end_forward, end_forward + lookback)
        ]
        dates.reverse()
        return dates

    def _merge_recent_rows(self, code: str, df: pd.DataFrame, days: int, now: datetime.datetime) -> bool:
        if not self._has_datetime_column(df, code, 'ts merge'):
            return False

        cache_df = self._ensure_code_loaded(code)
        if not self._has_datetime_column(cache_df, code, 'cache append'):
            self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
            cache_df = self[code]

        updated = False
        forward_end = -1 if now.hour > 18 else 0
        for forward_day in range(days, forward_end, -1):
            target_date_int = int(get_prev_trading_date(now, forward_day))
            target_date_df = df[df['datetime'] == target_date_int]
            if len(target_date_df) == 1 and (not (cache_df['datetime'] == target_date_int).any()):
                updated = True
                if self.cache_history[code] is None or len(self.cache_history[code]) == 0:
                    self.cache_history[code] = target_date_df
                else:
                    self.cache_history[code] = pd.concat(
                        [self.cache_history[code], target_date_df],
                        ignore_index=True,
                    )
                cache_df = self.cache_history[code]
        return updated

    def _split_batch_and_fallback(
        self,
        group_codes: list[str],
        start_date: str,
        end_date: str,
    ) -> tuple[list[tuple[str, pd.DataFrame]], list[str]]:
        """小批量拉取；返回 (批内成功, 需逐股 fallback)。"""
        batch_dfs = get_ts_daily_histories_flat(
            group_codes,
            start_date,
            end_date,
            columns=self.default_columns,
            interval=self.ts_download_interval,
        )

        batch_ok: list[tuple[str, pd.DataFrame]] = []
        need_single: list[str] = []

        if len(batch_dfs) == 0:
            return [], list(group_codes)

        for code in group_codes:
            code_df = batch_dfs.get(code)
            if code_df is not None and len(code_df) > 0 and self._has_datetime_column(code_df, code, 'ts batch'):
                batch_ok.append((code, code_df))
            else:
                need_single.append(code)
        return batch_ok, need_single

    def _process_incremental_code(
        self,
        code: str,
        code_df: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
        days: int,
        now: datetime.datetime,
        fail_codes: list[str],
    ) -> tuple[bool, bool]:
        """处理单股增量数据，返回 (成功, 是否有新行合并)。"""
        if code_df is None or len(code_df) == 0:
            fail_codes.append(code)
            return False, False

        code_df, err = self._prepare_code_df_for_save(code, code_df, start_date, end_date)
        if code_df is None:
            fail_codes.append(code)
            return False, False

        merged = self._merge_recent_rows(code, code_df, days, now)
        return True, merged

    def _process_single_daily_code(
        self,
        code: str,
        code_df: Optional[pd.DataFrame],
        target_date_int: int,
        fail_codes: list[str],
    ) -> tuple[bool, bool]:
        """处理单日补数据，返回 (成功, 是否有新行合并)。"""
        if code_df is None or len(code_df) != 1:
            fail_codes.append(code)
            return False, False
        if not self._has_datetime_column(code_df, code, 'ts single day'):
            fail_codes.append(code)
            return False, False

        cache_df = self._ensure_code_loaded(code)
        if not self._has_datetime_column(cache_df, code, 'cache append'):
            self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
            cache_df = self[code]
        if (cache_df['datetime'] == target_date_int).any():
            return True, False

        if len(cache_df) == 0:
            self.cache_history[code] = code_df
        else:
            self.cache_history[code] = pd.concat([cache_df, code_df], ignore_index=True)
        return True, True

    def _update_codes_in_batches(self, days: int, code_list: list[str]) -> set[str]:
        now = datetime.datetime.now()
        end_forward = self._incremental_end_forward(now)
        start_date = get_prev_trading_date(now, days + end_forward - 1)
        end_date = get_prev_trading_date(now, end_forward)

        need_update, skipped = self._filter_codes_need_recent_update(code_list, days, now)
        trading_dates = self._recent_trading_date_ints(now, days)
        check_range = (
            f'{trading_dates[0]}-{trading_dates[-1]}'
            if trading_dates
            else f'{start_date}-{end_date}'
        )
        print(
            f'{self._log_prefix} 增量更新 {check_range}（近 {days} 个交易日），'
            f'跳过 {skipped} 只，待更新 {len(need_update)} 只，'
            f'批量 {self.ts_batch_group_size} 只/批',
            flush=True,
        )
        if not need_update:
            return set()

        updated_codes: set[str] = set()
        updated_count = 0
        fail_codes: list[str] = []
        total = len(need_update)
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for i in range(0, len(need_update), self.ts_batch_group_size):
            group_codes = need_update[i:i + self.ts_batch_group_size]
            batch_ok, need_single = self._split_batch_and_fallback(group_codes, start_date, end_date)

            for code, code_df in batch_ok:
                ok, merged = self._process_incremental_code(
                    code, code_df, start_date, end_date, days, now, fail_codes,
                )
                scanned += 1
                if ok:
                    success_count += 1
                    if merged:
                        updated_codes.add(code)
                        updated_count += 1
                self._emit_scanned_progress('.' if ok else 'x', success_count, scanned)

            for code in need_single:
                time.sleep(self.ts_download_interval)
                code_df = self._fetch_single_code(code, start_date, end_date)
                ok, merged = self._process_incremental_code(
                    code, code_df, start_date, end_date, days, now, fail_codes,
                )
                scanned += 1
                if ok:
                    success_count += 1
                    if merged:
                        updated_codes.add(code)
                        updated_count += 1
                self._emit_scanned_progress('.' if ok else 'x', success_count, scanned)

        fail_count = len(fail_codes)
        print(
            f'\n{self._log_prefix} 增量完成 {updated_count}/{total} 更新 '
            f'{success_count}/{total} 成功 '
            f'{fail_count}/{total} 失败',
            flush=True,
        )
        if fail_codes:
            print(f'{self._log_prefix} 失败: {fail_codes}', flush=True)
        return updated_codes

    @staticmethod
    def _print_save_milestone(index: int) -> None:
        print(f'.{index}', end='', flush=True)

    def _sort_and_save_updated_codes(self, updated_codes: set[str]) -> None:
        total = len(updated_codes)
        if total == 0:
            print(f'{self._log_prefix} 无需落盘')
            return

        print(f'{self._log_prefix} 排序落盘中', end='', flush=True)
        end_forward = self._incremental_end_forward(datetime.datetime.now())
        expected_end = get_prev_trading_date(datetime.datetime.now(), end_forward)
        saved = 0
        skipped = 0
        for i, code in enumerate(updated_codes, 1):
            if i % 1000 == 0:
                self._print_save_milestone(i)
            if not self._has_datetime_column(self[code], code, 'save'):
                continue
            if not self._has_end_date_bar(self[code], expected_end):
                if not self._accept_suspend_tail(self[code], expected_end):
                    skipped += 1
                    continue
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(
                f'{self.root_path}/{self.default_kline_folder}/{code}.csv',
                index=False,
            )
            saved += 1

        if total % 1000 != 0:
            self._print_save_milestone(total)
        print(f'\n{self._log_prefix} 落盘完成 {saved} 只，跳过 {skipped} 只（尾部不齐）', flush=True)

    def download_single_daily(self, target_date: str) -> None:
        if len(self.cache_history) == 0 and not self._disk_last_datetime:
            self._prepare_incremental_cache()

        target_date_int = int(target_date)
        code_list = self.get_code_list()
        loss_list: list[str] = []
        for code in code_list:
            if self._code_has_trading_dates(code, [target_date_int]):
                continue
            loss_list.append(code)

        print(
            f'{self._log_prefix} 单日更新 {target_date}，待补 {len(loss_list)} 只，'
            f'批量 {self.ts_batch_group_size} 只/批',
        )
        if not loss_list:
            return

        updated_codes: set[str] = set()
        fail_codes: list[str] = []
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for i in range(0, len(loss_list), self.ts_batch_group_size):
            group_codes = loss_list[i:i + self.ts_batch_group_size]
            batch_ok, need_single = self._split_batch_and_fallback(
                group_codes, target_date, target_date,
            )

            for code, code_df in batch_ok:
                ok, merged = self._process_single_daily_code(code, code_df, target_date_int, fail_codes)
                scanned += 1
                if ok:
                    success_count += 1
                    if merged:
                        updated_codes.add(code)
                self._emit_scanned_progress('.' if ok else 'x', success_count, scanned)

            for code in need_single:
                time.sleep(self.ts_download_interval)
                code_df = self._fetch_single_code(code, target_date, target_date)
                ok, merged = self._process_single_daily_code(code, code_df, target_date_int, fail_codes)
                scanned += 1
                if ok:
                    success_count += 1
                    if merged:
                        updated_codes.add(code)
                self._emit_scanned_progress('.' if ok else 'x', success_count, scanned)

        fail_count = len(fail_codes)
        print(
            f'\n{self._log_prefix} 单日完成 {len(updated_codes)}/{len(loss_list)} 更新 '
            f'{success_count}/{len(loss_list)} 成功 '
            f'{fail_count}/{len(loss_list)} 失败',
            flush=True,
        )
        if fail_codes:
            print(f'{self._log_prefix} 失败: {fail_codes}', flush=True)

        self._sort_and_save_updated_codes(updated_codes)

    def download_recent_daily(self, days: int) -> bool:
        if len(self.cache_history) == 0 and not self._disk_last_datetime:
            self._prepare_incremental_cache()

        try:
            self._download_remote_missed()
        except Exception as e:
            print(f'{self._log_prefix} _download_remote_missed 异常: {e}')
            traceback.print_exc()

        try:
            code_list = self.get_code_list()
        except Exception as e:
            print(f'{self._log_prefix} get_code_list 异常: {e}')
            traceback.print_exc()
            return False

        try:
            ttl = self.since_last_update_datetime()
            if ttl is not None and ttl < 12 * 3600:
                print(f'{self._log_prefix} 距上次更新仅 {int(ttl)}s，跳过重复执行')
                self.load_history_from_disk_to_memory(auto_update=False)
                return True

            all_updated_codes = self._update_codes_in_batches(days, code_list)
        except Exception as e:
            print(f'{self._log_prefix} _update_codes_in_batches({days}) 异常: {e}')
            traceback.print_exc()
            return False

        self._sort_and_save_updated_codes(all_updated_codes)

        self.write_last_update_datetime()
        self._print_recent_tail_validation(code_list, lookback=days)
        self.load_history_from_disk_to_memory(auto_update=False)
        return True
