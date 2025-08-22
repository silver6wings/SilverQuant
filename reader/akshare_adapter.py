from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime
from typing import Optional

from tools.utils_basic import is_stock, code_to_symbol
from .base_adapter import BaseAdapter
from .constants import DataSource, ReadOptions


class AkshareAdapter(BaseAdapter):
    """Adapter for Akshare data source."""
    source = DataSource.AKSHARE

    def fetch_daily_history(
        self,
        code: str,
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions
    ) -> Optional[pd.DataFrame]:
        """Fetch daily history data from Akshare.

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
            import akshare as ak

            df = ak.stock_zh_a_hist(
                symbol=code_to_symbol(code),
                start_date=start or "20100101",
                end_date=end or datetime.now().strftime("%Y%m%d"),
                adjust=options.adjust or "",
                period='daily',
            )

            # Rename columns to standard format
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
            })

            if len(df) > 0:
                # Convert datetime to YYYYMMDD format
                df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')
                # Select required columns
                return df[["datetime", "open", "high", "low", "close", "volume", "amount"]]

            return None

        except Exception as e:
            logging.debug(f"AkshareAdapter error: {e}")
            return None
