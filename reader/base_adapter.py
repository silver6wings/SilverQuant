from __future__ import annotations

import pandas as pd
from typing import Dict, List, Optional

from .constants import ReadOptions


class BaseAdapter:
    """Base class for all data source adapters."""
    source: str

    def fetch_daily_history(
        self,
        code: str,
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV history for a given code.

        Args:
            code: Stock code (e.g., '000001.SZ')
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        raise NotImplementedError

    def fetch_daily_history_batch(
        self,
        codes: List[str],
        start: Optional[str],
        end: Optional[str],
        options: ReadOptions,
    ) -> Optional[Dict[str, pd.DataFrame]]:
        """Fetch daily OHLCV history for multiple codes in batch.

        Default implementation falls back to individual fetch_daily_history calls.
        Subclasses can override this for more efficient batch processing.

        Args:
            codes: List of stock codes (e.g., ['000001.SZ', '000002.SZ'])
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type

        Returns:
            Dict mapping code to DataFrame, or None if batch not supported
        """
        # Default implementation: return None to indicate batch not supported
        return None

    def supports_batch(self) -> bool:
        """Check if this adapter supports batch fetching.

        Returns:
            True if batch fetching is supported, False otherwise
        """
        return False
