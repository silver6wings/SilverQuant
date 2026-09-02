import os
import datetime
import time
import traceback
import pandas as pd

from tools.utils_basic import symbol_to_code
from tools.utils_cache import (
    AKCacheProtected,
    get_prev_trading_date,
    get_prev_trading_date_list,
    get_recent_exit_right_codes_from_fhps,
)
from tools.utils_remote import DataSource, ExitRight, get_daily_history


DEFAULT_INIT_DAY_COUNT: int = 550   # 默认足够覆盖两年


class DailyHistoryCache:
    _instance = None
    daily_history = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DailyHistoryCache, cls).__new__(cls)
            cls.daily_history = None  # Initialize data as None initially
        return cls._instance

    def __init__(self):
        self.data_source = DailyHistory.default_data_source
        # init 之后一定set之后daily_history才不会是None

    def set_data_source(self, data_source: DataSource, init_day_count: int = DEFAULT_INIT_DAY_COUNT):
        if self.daily_history is None or self.data_source != data_source:
            self.data_source = data_source
            if self.data_source == DataSource.MINIQMT:
                from tools.utils_xtquant import warn_native_only
                warn_native_only("DailyHistoryXT")
                from delegate.daily_history_xt import DailyHistoryXT
                self.daily_history = DailyHistoryXT(init_day_count=init_day_count)
            elif self.data_source == DataSource.TUSHARE:
                from delegate.daily_history_ts import DailyHistoryTS
                self.daily_history = DailyHistoryTS(init_day_count=init_day_count)
            else:
                self.daily_history = DailyHistory(data_source=self.data_source, init_day_count=init_day_count)
            self.daily_history.load_history_from_disk_to_memory()


class DailyHistory:
    """通用历史日线缓存（MOOTDX / AKSHARE 等）。

    Tushare → DailyHistoryTS；miniQMT → DailyHistoryXT。
    磁盘 CSV 格式统一；Tushare 价格为不复权，且增量 18:59 后含当日（见 DailyHistoryTS）。
    """
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    default_root_path: str = '_cache/_daily'
    default_kline_folder: str = 'kline'
    default_data_source: DataSource = DataSource.MOOTDX
    # MOOTDX 数据源 不要超过800，700为安全

    def __init__(
        self,
        root_path: str = default_root_path,
        data_source: DataSource = default_data_source,
        init_day_count: int = DEFAULT_INIT_DAY_COUNT,
    ):
        self.root_path = f'{root_path}_{data_source}'
        self.data_source = data_source
        self.init_day_count = init_day_count
        self.last_update_time = f'{self.root_path}/_last_update_time.txt'

        os.makedirs(self.root_path, exist_ok=True)
        os.makedirs(f'{self.root_path}/{self.default_kline_folder}', exist_ok=True)
        self.cache_history: dict[str, pd.DataFrame] = {}
        self._disk_last_datetime: dict[str, int] = {}

    _log_prefix = '[历史日线]'
    tail_suspend_lookback = 3

    def __getitem__(self, item: str) -> pd.DataFrame:
        if item not in self.cache_history:
            self.cache_history[item] = pd.DataFrame(columns=self.default_columns)
        return self.cache_history[item]

    def _has_datetime_column(self, df: pd.DataFrame, code: str, stage: str) -> bool:
        if df is None:
            print(f'[历史日线] {stage} {code} skipped: dataframe is None')
            return False
        if not isinstance(df, pd.DataFrame):
            print(f'[历史日线] {stage} {code} skipped: invalid dataframe type {type(df)}')
            return False
        if 'datetime' not in df.columns:
            print(f'[历史日线] {stage} {code} skipped: missing datetime column')
            return False
        return True

    # 获取数据副本
    def get_subset_copy(self, codes: list[str], days: int) -> dict[str, pd.DataFrame]:
        if codes is None:
            codes = self.cache_history.keys()

        ans = {}
        i = 0
        for code in codes:
            if code in self.cache_history:
                i += 1
                ans[code] = self[code].tail(days).copy()
        print(f'[历史日线] Find {i}/{len(codes)} codes returned.')
        return ans

    # 获取代码列表
    def get_code_list(self, force_download: bool = False, prefixes: set[str] = None) -> list[str]:
        code_list_path = f'{self.root_path}/_code_list.csv'

        # 获取远程所有股票代码列表，目前只支持 akshare
        if force_download:
            try:
                df = AKCacheProtected.stock_info_a_code_name()
                df.to_csv(code_list_path, index=False)
            except Exception as e:
                print('[历史日线] Download code list failed! ', e)

        # 获取本地列表时 prefix 生效
        if os.path.exists(code_list_path):
            df = pd.read_csv(code_list_path, dtype={'code': str}, index_col=False)
            if prefixes is None:
                return [symbol_to_code(symbol) for symbol in df['code'].values]
            else:
                return [symbol_to_code(symbol) for symbol in df['code'].values if symbol[:2] in prefixes]

        else:
            return []

    # ==============
    #  磁盘 tail 扫描（增量缺失检测，子类复用）
    # ==============

    @staticmethod
    def _parse_csv_line_datetime(line: str) -> int | None:
        try:
            val = int(line.split(',')[0])
            if 19900101 <= val <= 21001231:
                return val
        except (ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _read_csv_last_datetime(path: str) -> int | None:
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return None
                chunk_size = min(size, 4096)
                f.seek(-chunk_size, os.SEEK_END)
                tail = f.read().decode('utf-8', errors='ignore')
        except OSError:
            return None

        lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        return DailyHistory._parse_csv_line_datetime(lines[-1])

    @classmethod
    def _read_csv_tail_datetimes(cls, path: str, max_rows: int = 8) -> list[int]:
        try:
            with open(path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size == 0:
                    return []
                chunk_size = min(size, max(4096, max_rows * 64))
                f.seek(-chunk_size, os.SEEK_END)
                tail = f.read().decode('utf-8', errors='ignore')
        except OSError:
            return []

        datetimes: list[int] = []
        for line in tail.splitlines():
            line = line.strip()
            if not line or line.startswith('datetime'):
                continue
            dt = cls._parse_csv_line_datetime(line)
            if dt is not None:
                datetimes.append(dt)
        return datetimes[-max_rows:]

    def _kline_dir(self) -> str:
        return f'{self.root_path}/{self.default_kline_folder}'

    def _codes_on_disk(self) -> set[str]:
        kline_dir = self._kline_dir()
        if not os.path.isdir(kline_dir):
            return set()
        return {
            name[:-4]
            for name in os.listdir(kline_dir)
            if name.endswith('.csv')
        }

    def _recent_trading_date_ints(self, now: datetime.datetime, lookback: int) -> list[int]:
        today = now.strftime('%Y-%m-%d')
        date_strs = list(get_prev_trading_date_list(today, lookback))
        if len(date_strs) == 0:
            return [int(get_prev_trading_date(now, 1))]
        return [int(str(d).replace('-', '')) for d in date_strs]

    def _code_has_trading_dates(self, code: str, trading_dates: list[int]) -> bool:
        if not trading_dates:
            return True

        cache_df = self.cache_history.get(code)
        if cache_df is not None and len(cache_df) > 0:
            if not self._has_datetime_column(cache_df, code, 'cache check'):
                return False
            have = set(cache_df['datetime'].astype(int).tolist())
            return all(d in have for d in trading_dates)

        path = f'{self._kline_dir()}/{code}.csv'
        if not os.path.isfile(path):
            return False

        tail_dates = self._read_csv_tail_datetimes(path, len(trading_dates) + 2)
        if not tail_dates:
            return False
        have = set(tail_dates)
        if all(d in have for d in trading_dates):
            return True

        self._disk_last_datetime[code] = tail_dates[-1]
        return False

    def _filter_codes_need_recent_update(
        self,
        code_list: list[str],
        days: int,
        now: datetime.datetime,
    ) -> tuple[list[str], int]:
        trading_dates = self._recent_trading_date_ints(now, days)
        need_update: list[str] = []
        skipped = 0
        for code in code_list:
            if self._code_has_trading_dates(code, trading_dates):
                skipped += 1
            else:
                need_update.append(code)
        return need_update, skipped

    def _prepare_incremental_cache(self) -> None:
        """增量更新前：扫描末交易日索引，不全量载入内存。"""
        t0 = time.monotonic()
        code_list = self.get_code_list()
        on_disk = self._codes_on_disk()
        self.cache_history.clear()
        self._disk_last_datetime.clear()

        total = len(code_list)
        indexed = 0
        print(f'{self._log_prefix} 正在索引', end='', flush=True)
        for i, code in enumerate(code_list, 1):
            if code in on_disk:
                last_dt = self._read_csv_last_datetime(f'{self._kline_dir()}/{code}.csv')
                if last_dt is not None:
                    self._disk_last_datetime[code] = last_dt
                    indexed += 1
            if i % 100 == 0:
                print('.', end='', flush=True)

        elapsed = time.monotonic() - t0
        missing = total - indexed
        print(
            f'\n{self._log_prefix} 索引完成 {indexed}/{total} 只 '
            f'({missing} 无本地csv) {elapsed:.1f}s',
            flush=True,
        )

    def _ensure_code_loaded(self, code: str) -> pd.DataFrame:
        cache_df = self.cache_history.get(code)
        if cache_df is not None and len(cache_df) > 0:
            return cache_df

        path = f'{self._kline_dir()}/{code}.csv'
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, dtype={'datetime': int})
                if self._has_datetime_column(df, code, 'load'):
                    self.cache_history[code] = df
                    return df
            except Exception:
                pass

        self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
        return self.cache_history[code]

    def _has_end_date_bar(self, code_df: pd.DataFrame, expected_end: str) -> bool:
        if code_df is None or len(code_df) == 0:
            return False
        return (code_df['datetime'] == int(expected_end)).any()

    def _last_datetime(self, code_df: pd.DataFrame) -> int | None:
        if code_df is None or len(code_df) == 0:
            return None
        return int(code_df['datetime'].max())

    def _accept_suspend_tail(self, code_df: pd.DataFrame, expected_end: str) -> bool:
        """末条 K 线在 expected_end 前若干交易日内 → 视作停牌等无 bar，允许保存。"""
        last_int = self._last_datetime(code_df)
        if last_int is None:
            return False

        end_int = int(expected_end)
        if last_int >= end_int:
            return True

        end_dt = datetime.datetime.strptime(expected_end, '%Y%m%d')
        floor_date = get_prev_trading_date(end_dt, self.tail_suspend_lookback)
        return last_int >= int(floor_date)

    def _print_recent_tail_validation(
        self,
        code_list: list[str],
        lookback: int | None = None,
        now: datetime.datetime | None = None,
    ) -> None:
        """检查最近 lookback 个交易日是否齐全，仅打印缺失项。"""
        now = now or datetime.datetime.now()
        days = lookback if lookback is not None else self.tail_suspend_lookback
        trading_dates = self._recent_trading_date_ints(now, days)
        if not trading_dates:
            return

        print(
            f'{self._log_prefix} 尾部校验：最近 {days} 个交易日 '
            f'{trading_dates[0]} ~ {trading_dates[-1]}',
        )
        incomplete: list[tuple[str, list[int]]] = []
        for code in code_list:
            cache_df = self.cache_history.get(code)
            if cache_df is not None and len(cache_df) > 0:
                if not self._has_datetime_column(cache_df, code, 'tail validate'):
                    incomplete.append((code, trading_dates))
                    continue
                have = set(cache_df['datetime'].astype(int).tolist())
                missing = [d for d in trading_dates if d not in have]
                if missing:
                    incomplete.append((code, missing))
                continue

            path = f'{self._kline_dir()}/{code}.csv'
            if not os.path.isfile(path):
                incomplete.append((code, trading_dates))
                continue
            tail_dates = self._read_csv_tail_datetimes(path, len(trading_dates) + 2)
            have = set(tail_dates)
            missing = [d for d in trading_dates if d not in have]
            if missing:
                incomplete.append((code, missing))

        if not incomplete:
            print(f'{self._log_prefix} 尾部校验通过：{len(code_list)} 只均齐全')
            return

        print(f'{self._log_prefix} 尾部不齐 {len(incomplete)}/{len(code_list)} 只：')
        for code, missing in incomplete:
            print(f'  {code} 缺 {missing}')

    @staticmethod
    def _print_progress_header(
        success: int,
        scanned: int,
        *,
        newline: bool = True,
        log_prefix: str = '[历史日线]',
    ) -> None:
        prefix = '\n' if newline else ''
        print(f'{prefix}{log_prefix} [{success}/{scanned}]', end='', flush=True)

    def _emit_scanned_progress(self, mark: str, success_count: int, scanned: int) -> None:
        print(mark, end='', flush=True)
        if scanned % 100 == 0:
            self._print_progress_header(success_count, scanned, log_prefix=self._log_prefix)

    # ==============
    #  内部下载代码
    # ==============

    def _download_single_code_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        interval: int,
        try_limit: int = 2,
    ) -> pd.DataFrame | None:
        """单股下载；成功即返回，仅在失败时重试。"""
        df = None
        for try_count in range(try_limit):
            try:
                df = get_daily_history(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    columns=self.default_columns,
                    adjust=ExitRight.QFQ,
                    data_source=self.data_source,
                )
                if df is not None and len(df) > 0:
                    return df
            except Exception:
                pass
            if try_count + 1 < try_limit:
                time.sleep(interval)
        return df

    def _save_downloaded_daily(self, code: str, df: pd.DataFrame) -> None:
        df.to_csv(f'{self.root_path}/{self.default_kline_folder}/{code}.csv', index=False)

    def _download_codes(self, code_list: list[str], day_count: int, interval: int = 5) -> None:
        now = datetime.datetime.now()
        forward_day = 1  # 不算今天
        start_date = get_prev_trading_date(now, forward_day + day_count)
        end_date = get_prev_trading_date(now, forward_day)

        downloaded_count = 0
        download_failure = []

        # Tushare 全量逻辑见 DailyHistoryTS._download_codes
        group_size = 10
        for i in range(0, len(code_list), group_size):
            group_codes = [sub_code for sub_code in code_list[i:i + group_size]]

            for code in group_codes:
                df = self._download_single_code_daily(code, start_date, end_date, interval)
                if df is None or len(df) == 0:
                    download_failure.append(code)
                    continue
                self._save_downloaded_daily(code, df)
                downloaded_count += 1

            print(f'[历史日线] [{downloaded_count}/{min(i + group_size, len(code_list))}]', group_codes)
        # 有可能是当天新股没有数据，下载失败也正常
        print(f'[历史日线] Download finished with {len(download_failure)} fails: {download_failure}')

    # 自动补全本地缺失股票代码
    def _download_local_missed(self):
        code_list = self.get_code_list()
        print(f'[历史日线] Checking local missed codes from {len(code_list)}...')
        missing_codes = []
        for code in code_list:
            path = f'{self.root_path}/{self.default_kline_folder}/{code}.csv'
            if not os.path.exists(path):
                missing_codes.append(code)

        print(f'[历史日线] Downloading missing {len(missing_codes)} codes...')
        self._download_codes(missing_codes, self.init_day_count)

    # 下载本地缺失的股票代码数据
    def _download_remote_missed(self) -> None:
        print('[历史日线] Searching local missed code...')
        prev_code_list = self.get_code_list()
        curr_code_list = self.get_code_list(force_download=True)
        gap_codes = []
        for code in curr_code_list:
            if code not in prev_code_list:
                gap_codes.append(code)
        print(f'[历史日线] Downloading {len(gap_codes)} gap codes data of {self.init_day_count} days...')
        self._download_codes(gap_codes, self.init_day_count)

    # ==============
    #  全量更新逻辑
    # ==============

    def load_history_from_disk_to_memory(self, auto_update: bool = True) -> None:
        code_list = self.get_code_list()
        if len(code_list) == 0:
            self.download_all_to_disk()
            code_list = self.get_code_list()

        if auto_update:
            self._download_local_missed()

        kline_dir = f'{self.root_path}/{self.default_kline_folder}'
        on_disk = {
            name[:-4]
            for name in os.listdir(kline_dir)
            if name.endswith('.csv')
        }

        print(
            f'[历史日线] Loading {len(code_list)} codes '
            f'({len(on_disk)} csv on disk)...',
            end='',
        )
        self.cache_history.clear()
        missing_count = 0
        error_count = 0
        loaded_count = 0
        missing_sample: list[str] = []
        for i, code in enumerate(code_list, 1):
            if i % 1000 == 0:
                print('.', end='', flush=True)
            if code not in on_disk:
                missing_count += 1
                if len(missing_sample) < 5:
                    missing_sample.append(code)
                continue
            path = f'{kline_dir}/{code}.csv'
            try:
                df = pd.read_csv(path, dtype={'datetime': int})
                if not self._has_datetime_column(df, code, 'load'):
                    error_count += 1
                    continue
                self.cache_history[code] = df
                loaded_count += 1
            except Exception:
                error_count += 1
        print(
            f'\n[历史日线] Loading finished: {loaded_count} loaded, '
            f'{missing_count} missing, {error_count} read errors',
        )
        if missing_count > 0:
            suffix = f' (e.g. {missing_sample})' if missing_sample else ''
            print(f'[历史日线] Missing csv: {missing_count}{suffix}')

    def download_all_to_disk(self, renew_code_list: bool = True, interval: int = 5) -> None:
        code_list = self.get_code_list(force_download=renew_code_list)
        print(f'[历史日线] Downloading all {len(code_list)} codes data of {self.init_day_count} days...')
        self._download_codes(code_list, self.init_day_count, interval=interval)

    # ==============
    #  部分更新逻辑
    # ==============

    def _update_codes_one_by_one(self, days: int, code_list: list[str]) -> set[str]:
        now = datetime.datetime.now()
        start_date = get_prev_trading_date(now, days)
        end_date = get_prev_trading_date(now, 1)

        need_update, skipped = self._filter_codes_need_recent_update(code_list, days, now)
        trading_dates = self._recent_trading_date_ints(now, days)
        check_range = (
            f'{trading_dates[0]}-{trading_dates[-1]}'
            if trading_dates
            else f'{start_date}-{end_date}'
        )
        print(
            f'{self._log_prefix} 增量更新 {check_range}（近 {days} 个交易日），'
            f'跳过 {skipped} 只，待更新 {len(need_update)} 只',
            end='',
            flush=True,
        )
        if not need_update:
            print(' 无需更新')
            return set()

        updated_codes = set()
        updated_count = 0
        group_size = 100
        for i in range(0, len(need_update), group_size):
            print(f'\n{self._log_prefix} [{min(i + group_size, len(need_update))}]', end='')
            group_codes = [sub_code for sub_code in need_update[i:i + group_size]]
            for code in group_codes:
                df = get_daily_history(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    columns=self.default_columns,
                    adjust=ExitRight.QFQ,
                    data_source=self.data_source,
                )
                if df is not None and len(df) > 0:
                    if not self._has_datetime_column(df, code, 'history update'):
                        print('x', end='')
                        continue
                    cache_df = self[code]
                    if not self._has_datetime_column(cache_df, code, 'cache append'):
                        self.cache_history[code] = pd.DataFrame(columns=self.default_columns)
                        cache_df = self[code]
                    updated = False
                    for forward_day in range(days, 0, -1):
                        target_date_int = int(get_prev_trading_date(now, forward_day))
                        target_date_df = df[df['datetime'] == target_date_int]
                        if len(target_date_df) == 1 and (not (cache_df['datetime'] == target_date_int).any()):
                            updated = True
                            if self.cache_history[code] is None or len(self.cache_history[code]) == 0:
                                self.cache_history[code] = target_date_df
                            else:
                                self.cache_history[code] = pd.concat(
                                    [self.cache_history[code], target_date_df], ignore_index=True)
                                cache_df = self.cache_history[code]
                    if updated:
                        updated_codes.add(code)
                        updated_count += 1
                    print('.', end='')
                else:
                    print('x', end='')
        print(f' {updated_count} codes updated!')
        return updated_codes

    # 平时手动操作补单日数据使用（Tushare 见 DailyHistoryTS）
    def download_single_daily(self, target_date: str) -> None:
        if len(self.cache_history) == 0 and not self._disk_last_datetime:
            self._prepare_incremental_cache()

        code_list = self.get_code_list()
        target_date_int = int(target_date)
        loss_list = [
            code for code in code_list
            if not self._code_has_trading_dates(code, [target_date_int])
        ]
        print(f'{self._log_prefix} 单日更新 {target_date}，待补 {len(loss_list)} 只')
        if not loss_list:
            return

        updated_codes: set[str] = set()
        fail_codes: list[str] = []
        success_count = 0
        scanned = 0
        self._print_progress_header(success_count, scanned, newline=False, log_prefix=self._log_prefix)

        for code in loss_list:
            df = get_daily_history(
                code=code,
                start_date=target_date,
                end_date=target_date,
                columns=self.default_columns,
                adjust=ExitRight.QFQ,
                data_source=self.data_source,
            )
            scanned += 1
            if df is None or len(df) != 1:
                fail_codes.append(code)
                self._emit_scanned_progress('x', success_count, scanned)
                continue
            if not self._has_datetime_column(df, code, 'single day update'):
                fail_codes.append(code)
                self._emit_scanned_progress('x', success_count, scanned)
                continue

            cache_df = self._ensure_code_loaded(code)
            if (cache_df['datetime'] == target_date_int).any():
                success_count += 1
                self._emit_scanned_progress('.', success_count, scanned)
                continue

            updated_codes.add(code)
            if len(cache_df) == 0:
                self.cache_history[code] = df
            else:
                self.cache_history[code] = pd.concat([cache_df, df], ignore_index=True)
            success_count += 1
            self._emit_scanned_progress('.', success_count, scanned)

        fail_count = len(fail_codes)
        print(
            f'\n{self._log_prefix} 单日完成 {len(updated_codes)}/{len(loss_list)} 更新 '
            f'{success_count}/{len(loss_list)} 成功 '
            f'{fail_count}/{len(loss_list)} 失败',
            flush=True,
        )
        if fail_codes:
            print(f'{self._log_prefix} 失败: {fail_codes}', flush=True)

        print(f'{self._log_prefix} Sort and Save all history data ', end='')
        i = 0
        for code in updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            if not self._has_datetime_column(self[code], code, 'save'):
                continue
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(
                f'{self.root_path}/{self.default_kline_folder}/{code}.csv',
                index=False,
            )
        print(f'\n{self._log_prefix} Finished with {i} files updated')

    # 更新近几日数据（Tushare 见 DailyHistoryTS）
    def download_recent_daily(self, days: int) -> bool:
        if len(self.cache_history) == 0 and not self._disk_last_datetime:
            self._prepare_incremental_cache()
        elif len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()

        try:
            self._download_remote_missed()
        except Exception as e:
            print(f'[历史日线] _download_remote_missed 异常: {e}')
            traceback.print_exc()

        try:
            code_list = self.get_code_list()
        except Exception as e:
            print(f'[历史日线] get_code_list 异常: {e}')
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

        # 排序存储所有更新过的数据
        print(f'{self._log_prefix} Sorting and Saving all history data ', end='')
        i = 0
        for code in all_updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            if not self._has_datetime_column(self[code], code, 'save'):
                continue
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(f'{self.root_path}/{self.default_kline_folder}/{code}.csv', index=False)
        print(f'\n{self._log_prefix} Finished with {i} files updated')

        self.write_last_update_datetime()
        self._print_recent_tail_validation(code_list, lookback=days)
        self.load_history_from_disk_to_memory(auto_update=False)
        return True

    def write_last_update_datetime(self):
        now = datetime.datetime.now()
        with open(self.last_update_time, 'w', encoding='utf-8') as f:
            f.write(now.isoformat())

    def since_last_update_datetime(self):
        try:
            with open(self.last_update_time, 'r', encoding='utf-8') as f:
                last_time_str = f.read().strip()

            last_time = datetime.datetime.fromisoformat(last_time_str)
            now = datetime.datetime.now()
            time_delta = now - last_time
            return time_delta.total_seconds()
        except Exception as e:
            print(f"Get local history update TTL failed: {str(e)}")
            return None

    # ==============
    #  除权更新逻辑
    # ==============

    def remove_single_history(self, code: str) -> bool:
        file_path = f'{self.root_path}/{self.default_kline_folder}/{code}.csv'
        try:
            if not os.path.isfile(file_path):
                return False
            os.remove(file_path)
            return True
        except PermissionError:
            print(f'[历史日线] No Permission deleting {file_path}')
            return False
        except OSError as e:
            print(f'[历史日线] Error when deleting: {e}')
            return False

    @staticmethod
    def get_recent_exit_right_codes(days: int) -> list[str]:
        return get_recent_exit_right_codes_from_fhps(days)

    def remove_recent_exit_right_histories(self, days: int) -> None:
        try:
            codes = self.get_recent_exit_right_codes(days)
        except Exception as e:
            print(f'[历史日线] 获取最近 {days} 天除权列表异常，跳过清理: {e}')
            return

        removed_count = 0
        for code in codes:
            try:
                if self.remove_single_history(code):
                    removed_count += 1
            except Exception as e:
                print(f'[历史日线] 删除 {code} 历史缓存失败: {e}')
        print(f'[历史日线] Removed {removed_count} histories with Exit Right announced')
