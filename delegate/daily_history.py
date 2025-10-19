import os
import datetime
import time
import random
import pandas as pd

from tools.utils_basic import symbol_to_code
from tools.utils_cache import AKCache, get_prev_trading_date
from tools.utils_remote import DataSource, ExitRight, get_daily_history, get_ts_daily_histories


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

    def set_data_source(self, data_source: DataSource):
        if self.daily_history is None or self.data_source != data_source:
            self.data_source = data_source
            self.daily_history = DailyHistory(data_source=self.data_source)
            self.daily_history.load_history_from_disk_to_memory()


class DailyHistory:
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    default_root_path: str = '_cache/_daily'
    default_data_source: DataSource = DataSource.MOOTDX
    # TUSHARE 数据源 不要超过8000，7000为安全
    # MOOTDX 数据源 不要超过800，700为安全
    default_init_day_count: int = 550   # 默认足够覆盖两年

    def __init__(
        self,
        root_path: str = default_root_path,
        data_source: DataSource = default_data_source,
        init_day_count: int = default_init_day_count,
    ):
        self.root_path = f'{root_path}_{data_source}'
        self.data_source = data_source
        self.init_day_count = init_day_count
        self.last_update_time = f'{self.root_path}/_last_update_time.txt'

        os.makedirs(self.root_path, exist_ok=True)
        self.cache_history: dict[str, pd.DataFrame] = {}

    def __getitem__(self, item: str) -> pd.DataFrame:
        if item not in self.cache_history:
            self.cache_history[item] = pd.DataFrame(columns=self.default_columns)
        return self.cache_history[item]

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
        print(f'[HISTORY] Find {i}/{len(codes)} codes returned.')
        return ans

    # 获取代码列表
    def get_code_list(self, force_download: bool = False, prefixes: set[str] = None) -> list[str]:
        code_list_path = f'{self.root_path}/_code_list.csv'

        # 获取远程所有股票代码列表，目前只支持 akshare
        if force_download:
            import akshare as ak
            try:
                df = AKCache.stock_info_a_code_name()
                df.to_csv(code_list_path, index=False)
            except Exception as e:
                print('[HISTORY] Download code list failed! ', e)

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
    #  内部下载代码
    # ==============

    def _download_codes(self, code_list: list[str], day_count: int) -> None:
        now = datetime.datetime.now()
        forward_day = 1  # 不算今天
        start_date = get_prev_trading_date(now, forward_day + day_count)
        end_date = get_prev_trading_date(now, forward_day)

        downloaded_count = 0
        download_failure = []

        group_size = 10
        for i in range(0, len(code_list), group_size):
            group_codes = [sub_code for sub_code in code_list[i:i + group_size]]

            for code in group_codes:
                df = get_daily_history(
                    code=code,
                    start_date=start_date,
                    end_date=end_date,
                    columns=self.default_columns,
                    adjust=ExitRight.QFQ,
                    data_source=self.data_source,
                )
                time.sleep(random.uniform(1, 5))
                if df is None or len(df) == 0:
                    download_failure.append(code)
                    continue
                else:
                    df.to_csv(f'{self.root_path}/{code}.csv', index=False)
                    downloaded_count += 1

            print(f'[HISTORY] [{downloaded_count}/{min(i + group_size, len(code_list))}]', group_codes)
        # 有可能是当天新股没有数据，下载失败也正常
        print(f'[HISTORY] Download finished with {len(download_failure)} fails: {download_failure}')

    # 自动补全本地缺失股票代码
    def _download_local_missed(self):
        code_list = self.get_code_list()
        print(f'[HISTORY] Checking local missed codes from {len(code_list)}...')
        missing_codes = []
        for code in code_list:
            path = f'{self.root_path}/{code}.csv'
            if not os.path.exists(path):
                missing_codes.append(code)

        print(f'[HISTORY] Downloading missing {len(missing_codes)} codes...')
        self._download_codes(missing_codes, self.init_day_count)

    # 下载本地缺失的股票代码数据
    def _download_remote_missed(self) -> None:
        print('[HISTORY] Searching local missed code...')
        prev_code_list = self.get_code_list()
        curr_code_list = self.get_code_list(force_download=True)
        gap_codes = []
        for code in curr_code_list:
            if code not in prev_code_list:
                gap_codes.append(code)
        print(f'[HISTORY] Downloading {len(gap_codes)} gap codes data of {self.init_day_count} days...')
        self._download_codes(gap_codes, self.init_day_count)

    # ==============
    #  全量更新逻辑
    # ==============

    def load_history_from_disk_to_memory(self, auto_update: bool = True) -> None:
        code_list = self.get_code_list()
        if len(code_list) == 0:
            self.download_all_to_disk()

        if auto_update:
            self._download_local_missed()

        print(f'[HISTORY] Loading {len(code_list)} codes...', end='')
        self.cache_history.clear()
        error_count = 0
        i = 0
        for code in code_list:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            path = f'{self.root_path}/{code}.csv'
            try:
                df = pd.read_csv(path, dtype={'datetime': int})
                self.cache_history[code] = df
            except Exception as e:
                print(code, e)
                error_count += 1
        print(f'\n[HISTORY] Loading finished with {error_count}/{i} errors')

    def download_all_to_disk(self, renew_code_list: bool = True) -> None:
        code_list = self.get_code_list(force_download=renew_code_list)
        print(f'[HISTORY] Downloading all {len(code_list)} codes data of {self.init_day_count} days...')
        self._download_codes(code_list, self.init_day_count)

    # ==============
    #  部分更新逻辑
    # ==============

    # 下载具体某天的数据 TUSHARE
    def _update_codes_by_tushare(self, target_date: str, code_list: list[str]) -> set[str]:
        target_date_int = int(target_date)
        print(f'[HISTORY] Updating {target_date} ', end='')

        loss_list = []  # 找到缺失当天数据的codes
        for code in code_list:
            if not (self[code]['datetime'] == target_date_int).any():
                loss_list.append(code)

        updated_codes = set()
        updated_count = 0
        group_size = 990
        for i in range(0, len(loss_list), group_size):
            time.sleep(0.5)
            group_codes = [sub_code for sub_code in loss_list[i:i + group_size]]

            dfs = get_ts_daily_histories(
                codes=group_codes,
                start_date=target_date,
                end_date=target_date,
                columns=self.default_columns,
                adjust=ExitRight.QFQ,
            )

            # 填补缺失的日期
            for code in dfs:
                df = dfs[code]
                if len(df) == 1 and (not (self[code]['datetime'] == target_date_int).any()):
                    updated_codes.add(code)
                    updated_count += 1
                    if self.cache_history[code] is None or len(self.cache_history[code]) == 0:
                        self.cache_history[code] = df  # concat len = 0 的 df 会报 warning
                    else:
                        self.cache_history[code] = pd.concat([self.cache_history[code], df], ignore_index=True)
            print('.', end='')
        print(f' {updated_count} codes updated!')
        return updated_codes

    # 下载一段时间的数据，逐个下载
    def _update_codes_one_by_one(self, days: int, code_list: list[str]) -> set[str]:
        now = datetime.datetime.now()
        start_date = get_prev_trading_date(now, days)
        end_date = get_prev_trading_date(now, 1)
        print(f'[HISTORY] Updating {start_date} - {end_date}', end='')

        updated_codes = set()
        updated_count = 0
        group_size = 100
        for i in range(0, len(code_list), group_size):
            print(f'\n[HISTORY] [{min(i + group_size, len(code_list))}]', end='')
            group_codes = [sub_code for sub_code in code_list[i:i + group_size]]
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
                    updated = False
                    for forward_day in range(days, 0, -1):
                        target_date_int = int(get_prev_trading_date(now, forward_day))
                        target_date_df = df[df['datetime'] == target_date_int]
                        if len(target_date_df) == 1 and (not (self[code]['datetime'] == target_date_int).any()):
                            updated = True
                            if self.cache_history[code] is None or len(self.cache_history[code]) == 0:
                                self.cache_history[code] = target_date_df  # concat len = 0 的 df 会报 warning
                            else:
                                self.cache_history[code] = pd.concat(
                                    [self.cache_history[code], target_date_df], ignore_index=True)
                    if updated:
                        updated_codes.add(code)
                        updated_count += 1
                    print('.', end='')
                else:
                    print('x', end='')
        print(f' {updated_count} codes updated!')
        return updated_codes

    # 平时手动操作补单日数据使用
    def download_single_daily(self, target_date: str) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()
        code_list = self.get_code_list()
        updated_codes = self._update_codes_by_tushare(target_date, code_list)
        print('[HISTORY] Sort and Save all history data ', end='')
        i = 0
        for code in updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(f'{self.root_path}/{code}.csv', index=False)
        print(f'\n[HISTORY] Finished with {i} files updated')

    # 更新近几日数据，不用全部下载，速度快也不容易被Ban IP
    def download_recent_daily(self, days: int) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()

        self._download_remote_missed()  # 先把之前的历史更新上，可能会有长度不够的问题

        ttl = self.since_last_update_datetime()
        if ttl is not None and ttl < 12 * 3600:   # 上次更新时间太近就不重复执行
            return

        code_list = self.get_code_list()

        # TUSHARE 支持一次下载多个票，AKSHARE & MOOTDX 只能全部扫描一遍
        if self.data_source == DataSource.TUSHARE:
            now = datetime.datetime.now()
            all_updated_codes = set()
            for forward_day in range(days, 0, -1):
                target_date = get_prev_trading_date(now, forward_day)
                sub_updated_codes = self._update_codes_by_tushare(target_date, code_list)
                all_updated_codes.update(sub_updated_codes)
        else:
            all_updated_codes = self._update_codes_one_by_one(days, code_list)

        # 排序存储所有更新过的数据
        print('[HISTORY] Sorting and Saving all history data ', end='')
        i = 0
        for code in all_updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(f'{self.root_path}/{code}.csv', index=False)
        print(f'\n[HISTORY] Finished with {i} files updated')

        self.write_last_update_datetime()

    def write_last_update_datetime(self):
        current_utc_time = datetime.datetime.utcnow()
        with open(self.last_update_time, 'w', encoding='utf-8') as f:
            f.write(current_utc_time.isoformat())

    def since_last_update_datetime(self):
        try:
            with open(self.last_update_time, 'r', encoding='utf-8') as f:
                last_time_str = f.read().strip()
            last_time = datetime.datetime.fromisoformat(last_time_str)
            current_time = datetime.datetime.utcnow()
            time_delta = current_time - last_time
            return time_delta.total_seconds()
        except Exception as e:
            print(f"Get local history update TTL failed: {str(e)}")
            return None

    # ==============
    #  除权更新逻辑
    # ==============

    def remove_single_history(self, code: str) -> bool:
        file_path = f'{self.root_path}/{code}.csv'
        try:
            if not os.path.isfile(file_path):
                return False
            os.remove(file_path)
            return True
        except PermissionError:
            print(f'[HISTORY] No Permission deleting {file_path}')
            return False
        except OSError as e:
            print(f'[HISTORY] Error when deleting: {e}')
            return False

    @staticmethod
    def get_recent_exit_right_codes(days: int) -> list[str]:
        import akshare as ak
        ans = []
        now = datetime.datetime.now()
        for i in range(days-1, -1, -1):
            date_str = get_prev_trading_date(now, i)
            df = ak.news_trade_notify_dividend_baidu(date=date_str)
            if df is not None and len(df) > 0 and '股票代码' in df.columns:
                codes = [symbol_to_code(symbol) for symbol in df['股票代码'].values if len(symbol) == 6]
                ans += codes
        return ans

    def remove_recent_exit_right_histories(self, days: int) -> None:
        codes = self.get_recent_exit_right_codes(days)
        removed_count = 0
        for code in codes:
            if self.remove_single_history(code):
                removed_count += 1
        print(f'[HISTORY] Removed {removed_count} histories with Exit Right announced')
