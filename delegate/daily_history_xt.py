"""
miniQMT 专用历史日线缓存：继承 DailyHistory，仅覆盖下载/更新路径。

依赖 miniQMT 客户端在线；磁盘 CSV 格式与 DailyHistory 完全一致。
"""
import datetime
import os
import time
import traceback
from typing import Optional

import pandas as pd

from delegate.daily_history import DailyHistory
from tools.constants import ExitRight
from tools.utils_cache import get_prev_trading_date
from tools.utils_remote import DataSource
from tools.utils_remote_xt import (
    QMT_DOWNLOAD_GROUP_SIZE,
    ensure_qmt_client_ready,
    get_qmt_daily_histories,
    reset_qmt_download_log,
)


class DailyHistoryXT(DailyHistory):
    """QMT xtdata 优化版 DailyHistory：batch 冷启动 + 增量日常更新。"""

    default_data_source = DataSource.MINIQMT
    _log_prefix = '[历史日线XT]'
    xt_read_group_size = QMT_DOWNLOAD_GROUP_SIZE
    xt_adjust = ExitRight.QFQ

    def __init__(
        self,
        root_path: str = DailyHistory.default_root_path,
        init_day_count: int = 550,
    ):
        from tools.utils_xtquant import warn_native_only
        warn_native_only("DailyHistoryXT")
        super().__init__(
            root_path=root_path,
            data_source=DataSource.MINIQMT,
            init_day_count=init_day_count,
        )
        self._disk_last_datetime: dict[str, int] = {}

    def load_history_from_disk_to_memory(self, auto_update: bool = True) -> None:
        if auto_update:
            print(
                f'{self._log_prefix} 已跳过 auto_update 逐股补下载'
                '（缺文件请用 download_all_to_disk / download_recent_daily）',
            )
        super().load_history_from_disk_to_memory(auto_update=False)

    def _download_local_missed(self) -> None:
        """XT 不在 load 阶段逐股补下载，避免 QMT 串行阻塞。"""
        code_list = self.get_code_list()
        on_disk = self._codes_on_disk()
        missing = [code for code in code_list if code not in on_disk]
        if not missing:
            return
        print(
            f'{self._log_prefix} 本地缺 {len(missing)} 只 csv，'
            f'请运行 download_all_to_disk 或 download_recent_daily 补齐',
        )

    def _split_histories(self, df: pd.DataFrame, codes: list[str]) -> dict[str, pd.DataFrame]:
        if df is None or df.empty or "code" not in df.columns:
            return {}

        ans: dict[str, pd.DataFrame] = {}
        for code in codes:
            code_df = df[df["code"] == code]
            if len(code_df) == 0:
                continue
            part = code_df[self.default_columns].copy().reset_index(drop=True)
            if self._has_datetime_column(part, code, "xt split"):
                ans[code] = part
        return ans

    def _retry_fetch_code(
        self,
        code: str,
        start_date: str,
        end_date: str,
        incrementally: bool,
    ) -> Optional[pd.DataFrame]:
        df = get_qmt_daily_histories(
            [code],
            start_date,
            end_date,
            columns=None,
            adjust=self.xt_adjust,
            incrementally=incrementally,
            skip_download=True,
        )
        return self._split_histories(df, [code]).get(code)

    def _prepare_code_df_for_save(
        self,
        code: str,
        code_df: Optional[pd.DataFrame],
        start_date: str,
        end_date: str,
        incrementally: bool,
    ) -> tuple[Optional[pd.DataFrame], Optional[str]]:
        """校验末交易日；缺失则单股重试；仍缺则按停牌规则放行或拒绝。"""
        if self._has_end_date_bar(code_df, end_date):
            return code_df, None

        last_int = self._last_datetime(code_df)
        retried = self._retry_fetch_code(code, start_date, end_date, incrementally)
        if retried is not None and len(retried) > 0:
            code_df = retried

        if self._has_end_date_bar(code_df, end_date):
            return code_df, None

        if code_df is not None and len(code_df) > 0 and self._accept_suspend_tail(code_df, end_date):
            new_last = self._last_datetime(code_df)
            print(
                f"{code} 缺 {end_date}（末条 {new_last}），"
                f"视为停牌/无成交，允许保存",
                end='',
                flush=True,
            )
            return code_df, None

        if code_df is None or len(code_df) == 0:
            return None, "empty"
        return None, f"missing_tail(last={last_int}, need={end_date})"

    def _merge_recent_rows(self, code: str, df: pd.DataFrame, days: int, now: datetime.datetime) -> bool:
        if not self._has_datetime_column(df, code, "xt merge"):
            return False

        cache_df = self._ensure_code_loaded(code)
        if not self._has_datetime_column(cache_df, code, "cache append"):
            self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
            cache_df = self[code]

        updated = False
        for forward_day in range(days, 0, -1):
            target_date_int = int(get_prev_trading_date(now, forward_day))
            target_date_df = df[df["datetime"] == target_date_int]
            if len(target_date_df) == 1 and (not (cache_df["datetime"] == target_date_int).any()):
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

    def _process_single_daily_code(
        self,
        code: str,
        code_df: Optional[pd.DataFrame],
        target_date_int: int,
        fail_codes: list[str],
    ) -> tuple[bool, bool]:
        if code_df is None or len(code_df) != 1:
            fail_codes.append(code)
            return False, False
        if not self._has_datetime_column(code_df, code, 'xt single day'):
            fail_codes.append(code)
            return False, False

        cache_df = self._ensure_code_loaded(code)
        if (cache_df['datetime'] == target_date_int).any():
            return True, False

        if len(cache_df) == 0:
            self.cache_history[code] = code_df
        else:
            self.cache_history[code] = pd.concat([cache_df, code_df], ignore_index=True)
        return True, True

    def _fetch_and_save_codes(
        self,
        code_list: list[str],
        start_date: str,
        end_date: str,
        incrementally: bool,
    ) -> tuple[int, list[str]]:
        downloaded_count = 0
        download_failure: list[str] = []
        total = len(code_list)
        if total == 0:
            return 0, []

        if not ensure_qmt_client_ready():
            print(f'{self._log_prefix} QMT 未连接，中止下载')
            return 0, list(code_list)

        reset_qmt_download_log()
        print(f'{self._log_prefix} 冷启动落盘 {total} 只', flush=True)
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for i in range(0, len(code_list), self.xt_read_group_size):
            group_codes = code_list[i:i + self.xt_read_group_size]
            df = get_qmt_daily_histories(
                group_codes,
                start_date,
                end_date,
                columns=None,
                adjust=self.xt_adjust,
                incrementally=incrementally,
            )
            code_dfs = self._split_histories(df, group_codes)

            for code in group_codes:
                code_df = code_dfs.get(code)
                if code_df is None or len(code_df) == 0:
                    download_failure.append(code)
                    scanned += 1
                    self._emit_scanned_progress('x', success_count, scanned)
                    continue

                code_df, err = self._prepare_code_df_for_save(
                    code, code_df, start_date, end_date, incrementally,
                )
                if code_df is None:
                    download_failure.append(code)
                    scanned += 1
                    self._emit_scanned_progress('x', success_count, scanned)
                    continue

                code_df.to_csv(
                    f"{self.root_path}/{self.default_kline_folder}/{code}.csv",
                    index=False,
                )
                downloaded_count += 1
                success_count += 1
                scanned += 1
                self._emit_scanned_progress('.', success_count, scanned)

        fail_count = len(download_failure)
        print(
            f'\n{self._log_prefix} 落盘完成 {downloaded_count}/{total} 成功'
            + (f' {fail_count} 失败' if fail_count else ''),
            flush=True,
        )
        return downloaded_count, download_failure

    def _download_remote_missed(self) -> None:
        print(f'{self._log_prefix} 检查 code_list 新增代码...')
        prev_code_list = self.get_code_list()
        curr_code_list = self.get_code_list(force_download=True)
        gap_codes = [code for code in curr_code_list if code not in prev_code_list]
        if not gap_codes:
            print(f'{self._log_prefix} 无新增代码')
            return
        print(f'{self._log_prefix} 下载新增 {len(gap_codes)} 只 {self.init_day_count} 日历史...')
        self._download_codes(gap_codes, self.init_day_count)

    def _download_codes(self, code_list: list[str], day_count: int, interval: int = 5) -> None:
        if not code_list:
            return

        now = datetime.datetime.now()
        forward_day = 1
        start_date = get_prev_trading_date(now, forward_day + day_count)
        end_date = get_prev_trading_date(now, forward_day)

        print(
            f'{self._log_prefix} 冷启动下载 {len(code_list)} 只 '
            f"({start_date}-{end_date})，分组大小 {self.xt_read_group_size}",
        )
        _, download_failure = self._fetch_and_save_codes(
            code_list,
            start_date,
            end_date,
            incrementally=False,
        )
        print(f'{self._log_prefix} Download finished with {len(download_failure)} fails: {download_failure}')

    def _update_codes_one_by_one(self, days: int, code_list: list[str]) -> set[str]:
        now = datetime.datetime.now()
        start_date = get_prev_trading_date(now, days)
        end_date = get_prev_trading_date(now, 1)

        need_update, skipped = self._filter_codes_need_recent_update(code_list, days, now)
        trading_dates = self._recent_trading_date_ints(now, days)
        check_range = (
            f"{trading_dates[0]}-{trading_dates[-1]}"
            if trading_dates
            else f"{start_date}-{end_date}"
        )
        print(
            f'{self._log_prefix} 增量更新 {check_range}（近 {days} 个交易日），'
            f"跳过 {skipped} 只，待更新 {len(need_update)} 只",
            flush=True,
        )
        if not need_update:
            return set()

        if not ensure_qmt_client_ready():
            print(f'{self._log_prefix} QMT 未连接，中止增量下载')
            return set()
        print(f'{self._log_prefix} QMT 客户端已连接', flush=True)

        reset_qmt_download_log()
        updated_codes: set[str] = set()
        updated_count = 0
        fail_codes: list[str] = []
        total = len(need_update)
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for i in range(0, len(need_update), self.xt_read_group_size):
            group_codes = need_update[i:i + self.xt_read_group_size]

            df = get_qmt_daily_histories(
                group_codes,
                start_date,
                end_date,
                columns=None,
                adjust=self.xt_adjust,
                incrementally=True,
            )
            code_dfs = self._split_histories(df, group_codes)

            for code in group_codes:
                code_df = code_dfs.get(code)
                if code_df is None or len(code_df) == 0:
                    fail_codes.append(code)
                    scanned += 1
                    self._emit_scanned_progress('x', success_count, scanned)
                    continue

                code_df, err = self._prepare_code_df_for_save(
                    code, code_df, start_date, end_date, incrementally=True,
                )
                if code_df is None:
                    fail_codes.append(code)
                    scanned += 1
                    self._emit_scanned_progress('x', success_count, scanned)
                    continue

                merged = self._merge_recent_rows(code, code_df, days, now)
                if merged:
                    updated_codes.add(code)
                    updated_count += 1
                success_count += 1
                scanned += 1
                self._emit_scanned_progress('.', success_count, scanned)

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
        expected_end = get_prev_trading_date(datetime.datetime.now(), 1)
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
        loss_list = [
            code for code in code_list
            if not self._code_has_trading_dates(code, [target_date_int])
        ]
        print(
            f'{self._log_prefix} 单日更新 {target_date}，待补 {len(loss_list)} 只，'
            f'批量 {self.xt_read_group_size} 只/批',
        )
        if not loss_list:
            return

        if not ensure_qmt_client_ready():
            print(f'{self._log_prefix} QMT 未连接，中止单日下载')
            return

        updated_codes: set[str] = set()
        fail_codes: list[str] = []
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for i in range(0, len(loss_list), self.xt_read_group_size):
            group_codes = loss_list[i:i + self.xt_read_group_size]
            df = get_qmt_daily_histories(
                group_codes,
                target_date,
                target_date,
                columns=None,
                adjust=self.xt_adjust,
            )
            code_dfs = self._split_histories(df, group_codes)

            for code in group_codes:
                code_df = code_dfs.get(code)
                ok, merged = self._process_single_daily_code(
                    code, code_df, target_date_int, fail_codes,
                )
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

            all_updated_codes = self._update_codes_one_by_one(days, code_list)
        except Exception as e:
            print(f'{self._log_prefix} _update_codes_one_by_one({days}) 异常: {e}')
            traceback.print_exc()
            return False

        self._sort_and_save_updated_codes(all_updated_codes)

        self.write_last_update_datetime()
        self._print_recent_tail_validation(code_list, lookback=days)
        self.load_history_from_disk_to_memory(auto_update=False)
        return True
