from __future__ import annotations

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from .base_adapter import BaseAdapter
from .constants import DataSource, ReadOptions


class XtdataAdapter(BaseAdapter):
    """Adapter for Xtdata data source."""
    source = DataSource.XTDATA

    def fetch_daily_history(
        self,
        code: str,
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions
    ) -> Optional[pd.DataFrame]:
        """Fetch daily history data from Xtdata.

        Args:
            code: Stock code (e.g., '000001.SZ')
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            from xtquant import xtdata
        except Exception as e:
            logging.debug(f"XtdataAdapter unavailable: {e}")
            return None

        try:
            start_time = start or "20100101"
            end_time = end or datetime.now().strftime("%Y%m%d")
            dividend_type = {"qfq": "front", "hfq": "back"}.get(options.adjust or "", "none")

            # get_market_data_ex returns dict[stock_code] -> DataFrame
            raw = xtdata.get_market_data_ex(
                stock_list=[code],
                period="1d",
                start_time=start_time,
                end_time=end_time,
                dividend_type=dividend_type,
                fill_data=True,
            )

            if not raw or code not in raw:
                return None

            df = raw[code]
            if df is None or len(df) == 0:
                return None

            # 处理时间戳格式的时间索引
            df = self._xtdata_to_standard(df)
            return df

        except Exception as e:
            logging.debug(f"XtdataAdapter error: {e}")
            return None

    def _xtdata_to_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理 xtdata 返回的时间戳格式数据，确保与 其他Adapter 格式一致"""
        df = df.copy()

        # 处理时间信息，转换为 YYYYMMDD 格式的 datetime 列
        if 'time' in df.columns:
            # 如果有 time 列（时间戳格式）
            df['datetime'] = pd.to_datetime(df['time'], unit='ms').dt.strftime('%Y%m%d')
            df.drop('time', axis=1, inplace=True)

        # 重置索引为 RangeIndex，与 AkshareAdapter 保持一致
        df.reset_index(drop=True, inplace=True)

        # 确保包含必要的列
        #还有xtdata的df还有preclose
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA

        # 确保数据类型与 AkshareAdapter 一致
        df['datetime'] = df['datetime'].astype('object')  # 字符串类型
        for col in ['open', 'high', 'low', 'close', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').astype('int64')

        # 返回与 AkshareAdapter 相同的列顺序和格式
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    def supports_batch(self) -> bool:
        """XtdataAdapter supports batch fetching."""
        return True

    def fetch_daily_history_batch(
        self,
        codes: List[str],
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch daily history data for multiple codes from Xtdata in batch.

        Args:
            codes: List of stock codes (e.g., ['000001.SZ', '000002.SZ'])
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type

        Returns:
            Dict mapping code to DataFrame, or None if failed
        """
        try:
            from xtquant import xtdata
        except Exception as e:
            logging.debug(f"XtdataAdapter unavailable: {e}")
            return None

        try:
            start_time = start or "20100101"
            end_time = end or datetime.now().strftime("%Y%m%d")
            dividend_type = {"qfq": "front", "hfq": "back"}.get(options.adjust or "", "none")

            # get_market_data_ex returns dict[stock_code] -> DataFrame
            raw = xtdata.get_market_data_ex(
                stock_list=codes,  # Pass all codes at once
                period="1d",
                start_time=start_time,
                end_time=end_time,
                dividend_type=dividend_type,
                fill_data=True,
            )

            if not raw:
                return None

            # Process results for each code
            result = {}
            for code in codes:
                if code in raw and raw[code] is not None and len(raw[code]) > 0:
                    # 处理时间戳格式
                    processed_df = self._xtdata_to_standard(raw[code])
                    result[code] = processed_df

            return result

        except Exception as e:
            logging.debug(f"XtdataAdapter batch error: {e}")
            return None
