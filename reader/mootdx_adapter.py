from __future__ import annotations

import logging
import pandas as pd
from typing import Optional

from .base_adapter import BaseAdapter
from .constants import DataSource, ReadOptions


class MootdxAdapter(BaseAdapter):
    """Adapter for Mootdx data source."""
    source = DataSource.MOOTDX

    def fetch_daily_history(
        self, 
        code: str, 
        start: Optional[str], 
        end: Optional[str], 
        options: ReadOptions
    ) -> Optional[pd.DataFrame]:
        """Fetch daily history data from Mootdx.
        
        Args:
            code: Stock code (e.g., '000001.SZ')
            start: Start date in YYYYMMDD format
            end: End date in YYYYMMDD format
            options: Read options including adjust type
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        logging.debug("MootdxAdapter not implemented yet")
        return None
