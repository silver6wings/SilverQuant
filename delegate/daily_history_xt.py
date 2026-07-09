"""
miniQMT 专用历史日线缓存：继承 DailyHistory，仅覆盖下载/更新路径。

依赖 miniQMT 客户端在线；磁盘 CSV 格式与 DailyHistory 完全一致。
"""
import datetime
import traceback
from typing import Optional

import pandas as pd

from delegate.daily_history import DailyHistory
from tools.constants import ExitRight
from tools.utils_cache import get_prev_trading_date, get_prev_trading_date_list
from tools.utils_remote import DataSource
from tools.utils_remote_xt import ensure_qmt_daily_downloaded, get_qmt_daily_histories


class DailyHistoryXT(DailyHistory):
    """QMT xtdata 优化版 DailyHistory：batch2 冷启动 + 增量日常更新。"""

    default_data_source = DataSource.MINIQMT
    xt_read_group_size = 100
    xt_adjust = ExitRight.QFQ
    # 缺目标末日时：末条 K 线仍落在此窗口内 → 视为停牌/无成交，允许落盘
    tail_suspend_lookback = 3

    def __init__(
        self,
        root_path: str = DailyHistory.default_root_path,
        init_day_count: int = 550,
    ):
        super().__init__(
            root_path=root_path,
            data_source=DataSource.MINIQMT,
            init_day_count=init_day_count,
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

    def _has_end_date_bar(self, code_df: pd.DataFrame, expected_end: str) -> bool:
        if code_df is None or len(code_df) == 0:
            return False
        return (code_df["datetime"] == int(expected_end)).any()

    def _last_datetime(self, code_df: pd.DataFrame) -> Optional[int]:
        if code_df is None or len(code_df) == 0:
            return None
        return int(code_df["datetime"].max())

    def _accept_suspend_tail(self, code_df: pd.DataFrame, expected_end: str) -> bool:
        """末条 K 线在 expected_end 前若干交易日内 → 视作停牌等无 bar，允许保存。"""
        last_int = self._last_datetime(code_df)
        if last_int is None:
            return False

        end_int = int(expected_end)
        if last_int >= end_int:
            return True

        end_dt = datetime.datetime.strptime(expected_end, "%Y%m%d")
        floor_date = get_prev_trading_date(end_dt, self.tail_suspend_lookback)
        return last_int >= int(floor_date)

    def _retry_fetch_code(
        self,
        code: str,
        start_date: str,
        end_date: str,
        incrementally: bool,
    ) -> Optional[pd.DataFrame]:
        ensure_qmt_daily_downloaded(
            [code],
            "" if incrementally else start_date,
            end_date,
            incrementally=True if incrementally else False,
        )
        df = get_qmt_daily_histories(
            [code],
            start_date,
            end_date,
            columns=None,
            adjust=self.xt_adjust,
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
                f"[历史日线XT] {code} 缺 {end_date}（末条 {new_last}），"
                f"视为停牌/无成交，允许保存",
            )
            return code_df, None

        if code_df is None or len(code_df) == 0:
            return None, "empty"
        return None, f"missing_tail(last={last_int}, need={end_date})"

    def _recent_trading_date_ints(self, now: datetime.datetime, lookback: int) -> list[int]:
        today = now.strftime("%Y-%m-%d")
        date_strs = list(get_prev_trading_date_list(today, lookback))
        if len(date_strs) == 0:
            return [int(get_prev_trading_date(now, 1))]
        return [int(str(d).replace("-", "")) for d in date_strs]

    def _cache_has_recent_end(self, code: str, expected_end: str) -> bool:
        cache_df = self[code]
        if not self._has_datetime_column(cache_df, code, "cache check"):
            return False
        return self._has_end_date_bar(cache_df, expected_end)

    def _filter_codes_need_recent_update(self, code_list: list[str], expected_end: str) -> tuple[list[str], int]:
        need_update: list[str] = []
        skipped = 0
        for code in code_list:
            if self._cache_has_recent_end(code, expected_end):
                skipped += 1
            else:
                need_update.append(code)
        return need_update, skipped

    def _print_recent_tail_validation(self, code_list: list[str], now: Optional[datetime.datetime] = None) -> None:
        """检查最近 tail_suspend_lookback 个交易日是否齐全，仅打印缺失项。"""
        now = now or datetime.datetime.now()
        trading_dates = self._recent_trading_date_ints(now, self.tail_suspend_lookback)
        if not trading_dates:
            return

        print(
            f"[历史日线XT] 尾部校验：最近 {self.tail_suspend_lookback} 个交易日 "
            f"{trading_dates[0]} ~ {trading_dates[-1]}",
        )
        incomplete: list[tuple[str, list[int]]] = []
        for code in code_list:
            cache_df = self[code]
            if not self._has_datetime_column(cache_df, code, "tail validate"):
                incomplete.append((code, trading_dates))
                continue
            have = set(cache_df["datetime"].astype(int).tolist())
            missing = [d for d in trading_dates if d not in have]
            if missing:
                incomplete.append((code, missing))

        if not incomplete:
            print(f"[历史日线XT] 尾部校验通过：{len(code_list)} 只均齐全")
            return

        print(f"[历史日线XT] 尾部不齐 {len(incomplete)}/{len(code_list)} 只：")
        for code, missing in incomplete:
            print(f"  {code} 缺 {missing}")

    def _merge_recent_rows(self, code: str, df: pd.DataFrame, days: int, now: datetime.datetime) -> bool:
        if not self._has_datetime_column(df, code, "xt merge"):
            return False

        cache_df = self[code]
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

    def _fetch_and_save_codes(
        self,
        code_list: list[str],
        start_date: str,
        end_date: str,
        incrementally: bool,
    ) -> tuple[int, list[str]]:
        downloaded_count = 0
        download_failure: list[str] = []

        for i in range(0, len(code_list), self.xt_read_group_size):
            group_codes = code_list[i:i + self.xt_read_group_size]
            ensure_qmt_daily_downloaded(
                group_codes,
                start_date,
                end_date,
                incrementally=incrementally,
            )
            df = get_qmt_daily_histories(
                group_codes,
                start_date,
                end_date,
                columns=None,
                adjust=self.xt_adjust,
            )
            code_dfs = self._split_histories(df, group_codes)

            for code in group_codes:
                code_df = code_dfs.get(code)
                if code_df is None or len(code_df) == 0:
                    download_failure.append(code)
                    continue

                code_df, err = self._prepare_code_df_for_save(
                    code, code_df, start_date, end_date, incrementally,
                )
                if code_df is None:
                    print(f"[历史日线XT] {code} 跳过落盘: {err}")
                    download_failure.append(code)
                    continue

                code_df.to_csv(
                    f"{self.root_path}/{self.default_kline_folder}/{code}.csv",
                    index=False,
                )
                downloaded_count += 1

            print(
                f"[历史日线XT] [{downloaded_count}/{min(i + self.xt_read_group_size, len(code_list))}]",
                group_codes,
            )

        return downloaded_count, download_failure

    def _download_codes(self, code_list: list[str], day_count: int, interval: int = 5) -> None:
        now = datetime.datetime.now()
        forward_day = 1
        start_date = get_prev_trading_date(now, forward_day + day_count)
        end_date = get_prev_trading_date(now, forward_day)

        print(
            f"[历史日线XT] 冷启动 batch2 下载 {len(code_list)} 只 "
            f"({start_date}-{end_date})，interval 参数对 QMT 无效已忽略",
        )
        _, download_failure = self._fetch_and_save_codes(
            code_list,
            start_date,
            end_date,
            incrementally=False,
        )
        print(f"[历史日线XT] Download finished with {len(download_failure)} fails: {download_failure}")

    def _update_codes_one_by_one(self, days: int, code_list: list[str]) -> set[str]:
        now = datetime.datetime.now()
        start_date = get_prev_trading_date(now, days)
        end_date = get_prev_trading_date(now, 1)

        need_update, skipped = self._filter_codes_need_recent_update(code_list, end_date)
        print(
            f"[历史日线XT] 增量更新 {start_date} - {end_date}，"
            f"跳过已有 {end_date} 的 {skipped} 只，待更新 {len(need_update)} 只",
            end="",
        )
        if not need_update:
            print(" 无需下载")
            return set()

        updated_codes: set[str] = set()
        updated_count = 0

        for i in range(0, len(need_update), self.xt_read_group_size):
            print(
                f"\n[历史日线XT] [{min(i + self.xt_read_group_size, len(need_update))}/"
                f"{len(need_update)}]",
                end="",
            )
            group_codes = need_update[i:i + self.xt_read_group_size]

            ensure_qmt_daily_downloaded(
                group_codes,
                "",
                end_date,
                incrementally=True,
            )
            df = get_qmt_daily_histories(
                group_codes,
                start_date,
                end_date,
                columns=None,
                adjust=self.xt_adjust,
            )
            code_dfs = self._split_histories(df, group_codes)

            for code in group_codes:
                code_df = code_dfs.get(code)
                if code_df is None or len(code_df) == 0:
                    print("x", end="")
                    continue

                code_df, err = self._prepare_code_df_for_save(
                    code, code_df, start_date, end_date, incrementally=True,
                )
                if code_df is None:
                    print("x", end="")
                    continue

                if self._merge_recent_rows(code, code_df, days, now):
                    updated_codes.add(code)
                    updated_count += 1
                    print(".", end="")

        print(f" {updated_count} codes updated!")
        return updated_codes

    def download_single_daily(self, target_date: str) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()

        target_date_int = int(target_date)
        code_list = self.get_code_list()
        loss_list: list[str] = []
        for code in code_list:
            cache_df = self[code]
            if not self._has_datetime_column(cache_df, code, "cache check"):
                self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
                cache_df = self[code]
            if not (cache_df["datetime"] == target_date_int).any():
                loss_list.append(code)

        print(f"[历史日线XT] Updating single day {target_date} for {len(loss_list)} codes")
        updated_codes: set[str] = set()

        for i in range(0, len(loss_list), self.xt_read_group_size):
            group_codes = loss_list[i:i + self.xt_read_group_size]
            ensure_qmt_daily_downloaded(
                group_codes,
                target_date,
                target_date,
                incrementally=False,
            )
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
                if code_df is None or len(code_df) != 1:
                    continue
                if not self._has_datetime_column(code_df, code, "xt single day"):
                    continue
                cache_df = self[code]
                if not self._has_datetime_column(cache_df, code, "cache append"):
                    self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
                    cache_df = self[code]
                if (cache_df["datetime"] == target_date_int).any():
                    continue
                updated_codes.add(code)
                if len(cache_df) == 0:
                    self.cache_history[code] = code_df
                else:
                    self.cache_history[code] = pd.concat([cache_df, code_df], ignore_index=True)

        print(f"[历史日线XT] Sort and Save all history data ", end="")
        expected_end = get_prev_trading_date(datetime.datetime.now(), 1)
        i = 0
        skipped = 0
        for code in updated_codes:
            i += 1
            if i % 1000 == 0:
                print(".", end="")
            if not self._has_datetime_column(self[code], code, "save"):
                continue
            if not self._has_end_date_bar(self[code], expected_end):
                if not self._accept_suspend_tail(self[code], expected_end):
                    skipped += 1
                    continue
            self.cache_history[code] = self[code].sort_values(by="datetime")
            self.cache_history[code].to_csv(
                f"{self.root_path}/{self.default_kline_folder}/{code}.csv",
                index=False,
            )
        print(f"\n[历史日线XT] Finished with {i - skipped} files updated, {skipped} skipped (tail)")

    def download_recent_daily(self, days: int) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory(auto_update=False)

        try:
            self._download_remote_missed()
        except Exception as e:
            print(f"[历史日线XT] _download_remote_missed 异常: {e}")
            traceback.print_exc()

        try:
            code_list = self.get_code_list()
        except Exception as e:
            print(f"[历史日线XT] get_code_list 异常: {e}")
            traceback.print_exc()
            return

        try:
            ttl = self.since_last_update_datetime()
            if ttl is not None and ttl < 12 * 3600:
                print(f"[历史日线XT] 距上次更新仅 {int(ttl)}s，跳过重复执行")
                return

            all_updated_codes = self._update_codes_one_by_one(days, code_list)
        except Exception as e:
            print(f"[历史日线XT] _update_codes_one_by_one({days}) 异常: {e}")
            traceback.print_exc()
            return

        print("[历史日线XT] Sorting and Saving all history data ", end="")
        expected_end = get_prev_trading_date(datetime.datetime.now(), 1)
        i = 0
        skipped = 0
        for code in all_updated_codes:
            i += 1
            if i % 1000 == 0:
                print(".", end="")
            if not self._has_datetime_column(self[code], code, "save"):
                continue
            if not self._has_end_date_bar(self[code], expected_end):
                if not self._accept_suspend_tail(self[code], expected_end):
                    skipped += 1
                    continue
            self.cache_history[code] = self[code].sort_values(by="datetime")
            self.cache_history[code].to_csv(
                f"{self.root_path}/{self.default_kline_folder}/{code}.csv",
                index=False,
            )
        print(f"\n[历史日线XT] Finished with {i - skipped} files updated, {skipped} skipped (tail)")

        self.write_last_update_datetime()
        self._print_recent_tail_validation(code_list)
