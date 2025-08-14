from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime
from typing import Optional

from tools.utils_basic import is_stock
from .base_adapter import BaseAdapter
from .constants import DataSource, ReadOptions


class TushareAdapter(BaseAdapter):
    """Adapter for Tushare data source."""
    source = DataSource.TUSHARE

    def fetch_daily_history(
        self,
        code: str,
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions
    ) -> Optional[pd.DataFrame]:
        """Fetch daily history data from Tushare.

        Args:
            code: Stock code (e.g., '000001.SZ')
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not is_stock(code):
            return None

        try:
            from reader.tushare_agent import get_tushare_pro

            try_times = 0
            df = None
            while (df is None or len(df) <= 0) and try_times < 3:
                pro = get_tushare_pro()
                try_times += 1
                df = pro.daily(
                    ts_code=code,
                    start_date=start or "20100101",
                    end_date=end or datetime.now().strftime("%Y%m%d"),
                )

            if df is not None and len(df) > 0:
                # Apply tushare standardization
                df = self._ts_to_standard(df)
                # Select required columns
                return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]

            return None

        except Exception as e:
            logging.debug(f"TushareAdapter error: {e}")
            return None

    def _ts_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert tushare format to standard format, ensuring consistency with AkshareAdapter."""
        df = df.rename(columns={
            'vol': 'volume',
            'trade_date': 'datetime',
        })

        # 确保 datetime 为字符串格式（与 AkshareAdapter 一致）
        df['datetime'] = df['datetime'].astype(str)

        # 确保数据类型与 AkshareAdapter 一致
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')

        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce') * 1000
        df['amount'] = df['amount'].round(2).astype('float64')

        # Reverse order and reset index to RangeIndex
        df = df[::-1]
        df.reset_index(drop=True, inplace=True)

        return df
