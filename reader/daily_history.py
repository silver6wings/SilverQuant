import os
import datetime

import pandas as pd

from tools.utils_basic import symbol_to_code
from tools.utils_cache import get_prev_trading_date
from tools.utils_remote import DataSource, get_daily_history, get_ts_daily_histories


class DailyHistoryCache:
    _instance = None
    daily_history = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DailyHistoryCache, cls).__new__(cls)
            cls.daily_history = None  # Initialize data as None initially
        return cls._instance

    def __init__(self):
        if self.daily_history is None:
            self.daily_history = DailyHistory()
            self.daily_history.load_history_from_disk_to_memory()


class DailyHistory:
    default_init_day_count: int = 550  # 不要超过8000，这个是tushare的上限，不是代码的上限
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']

    def __init__(self, root_path: str = './_cache/_daily', init_day_count: int = default_init_day_count):
        self.root_path = root_path
        self.init_day_count = init_day_count
        self.cache_history: dict[str, pd.DataFrame] = {}

    def __getitem__(self, item) -> pd.DataFrame:
        if item not in self.cache_history:
            self.cache_history[item] = pd.DataFrame(columns=self.default_columns)
        return self.cache_history[item]

    def get_subset_copy(self, codes: list[str], days: int) -> dict[str, pd.DataFrame]:
        if codes is None:
            codes = self.cache_history.keys()

        ans = {}
        i = 0
        for code in codes:
            if code in self.cache_history:
                i += 1
                ans[code] = self[code].tail(days).copy()
        print(f'Find {i}/{len(codes)} codes returned.')
        return ans

    def get_code_list(self, force_download: bool = False, prefixes: set[str] = None) -> list[str]:
        path = f'{self.root_path}/_code_list.csv'

        if force_download:
            import akshare as ak
            try:
                df = ak.stock_info_a_code_name()
                df.to_csv(path, index=False)
            except Exception as e:
                print('Download code list failed! ', e)

        if os.path.exists(path):
            df = pd.read_csv(path, dtype={'code': str}, index_col=False)
            if prefixes is None:
                return [symbol_to_code(symbol) for symbol in df['code'].values]
            else:
                return [symbol_to_code(symbol) for symbol in df['code'].values if symbol[:2] in prefixes]

        else:
            return []

    def load_history_from_disk_to_memory(
        self,
        data_source: DataSource = DataSource.TUSHARE,
    ) -> None:
        code_list = self.get_code_list()
        if len(code_list) == 0:
            self.download_all_to_disk()

        code_list = self.get_code_list()

        print(f'Checking {len(code_list)} codes...', end='')
        missing_codes = []
        for code in code_list:
            path = f'{self.root_path}/{code}.csv'
            if not os.path.exists(path):
                missing_codes.append(code)

        print(f'Downloading {len(missing_codes)} codes...')
        self._download_codes(missing_codes, self.init_day_count, data_source)

        print(f'Loading {len(code_list)} codes...', end='')
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
        print(f'\nLoading finished with {error_count}/{i} errors')

    def _download_codes(self, code_list: list[str], day_count: int, data_source: DataSource = DataSource.TUSHARE):
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
                    adjust='qfq',
                    data_source=data_source,
                )
                if df is None or len(df) == 0:
                    download_failure.append(code)
                    continue
                else:
                    df.to_csv(f'{self.root_path}/{code}.csv', index=False)
                    downloaded_count += 1
            print(f'[{downloaded_count}/{min(i + group_size, len(code_list))}]', group_codes)
        # 有可能是当天新股没有数据，下载失败也正常
        print(f'Download finished with {len(download_failure)} fails: {download_failure}')

    def download_all_to_disk(self, data_source: DataSource = DataSource.TUSHARE, renew_code_list: bool = True) -> None:
        code_list = self.get_code_list(force_download=renew_code_list)
        print(f'Downloading all {len(code_list)} codes data of {self.init_day_count} days...')
        self._download_codes(code_list, self.init_day_count, data_source)

    def _download_gap_to_disk(self, data_source: DataSource = DataSource.TUSHARE) -> None:
        prev_code_list = self.get_code_list()
        curr_code_list = self.get_code_list(force_download=True)
        gap_codes = []
        for code in curr_code_list:
            if code not in prev_code_list:
                gap_codes.append(code)
        print(f'Downloading {len(gap_codes)} gap codes data of {self.init_day_count} days...')
        self._download_codes(gap_codes, self.init_day_count, data_source)

    def _download_date(self, target_date: str, code_list: list[str]) -> set[str]:
        target_date_int = int(target_date)
        print(f'Updating {target_date} ', end='')

        loss_list = []
        for code in code_list:
            if not (self[code]['datetime'] == target_date_int).any():
                loss_list.append(code)

        updated_codes = set()
        updated_count = 0
        group_size = 1000
        for i in range(0, len(loss_list), group_size):
            group_codes = [sub_code for sub_code in loss_list[i:i + group_size]]

            dfs = get_ts_daily_histories(
                codes=group_codes,
                start_date=target_date,
                end_date=target_date,
                columns=self.default_columns,
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
        print(f' {updated_count} Updated!')
        return updated_codes

    def download_single_daily(self, target_date: str) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()

        # self._download_gap_to_disk()  # 平时手动操作补单日数据需要，这里就先注释掉

        code_list = self.get_code_list()
        updated_codes = self._download_date(target_date, code_list)
        print('Sort and Save all history data ', end='')
        i = 0
        for code in updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(f'{self.root_path}/{code}.csv', index=False)
        print(f'\nFinished with {i} files updated')

    def download_recent_daily(self, days: int) -> None:
        if len(self.cache_history) == 0:
            self.load_history_from_disk_to_memory()

        self._download_gap_to_disk()  # 先把之前的历史更新上，可能会有长度不够的问题

        code_list = self.get_code_list()

        now = datetime.datetime.now()
        all_updated_codes = set()
        for forward_day in range(days, 0, -1):
            target_date = get_prev_trading_date(now, forward_day)
            sub_updated_codes = self._download_date(target_date, code_list)
            all_updated_codes.update(sub_updated_codes)

        print('Sort and Save all history data ', end='')
        i = 0
        for code in all_updated_codes:
            i += 1
            if i % 1000 == 0:
                print('.', end='')
            self.cache_history[code] = self[code].sort_values(by='datetime')
            self.cache_history[code].to_csv(f'{self.root_path}/{code}.csv', index=False)
        print(f'\nFinished with {i} files updated')
