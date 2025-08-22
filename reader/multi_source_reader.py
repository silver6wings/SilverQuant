from __future__ import annotations

import logging
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Union

# Import adapters
from .akshare_adapter import AkshareAdapter
from .amazing_data_adapter import AmazingDataAdapter
from .base_adapter import BaseAdapter
# Import constants and utilities
from .constants import DataCategory, DataSource, ReadOptions
from .mootdx_adapter import MootdxAdapter
from .tushare_adapter import TushareAdapter
from .utils import CodesLike, _ensure_list, _normalize_date
from .xtdata_adapter import XtdataAdapter

# Registry of adapters in default priority order (after user-preferred)
DEFAULT_ADAPTERS: List[BaseAdapter] = [
    XtdataAdapter(),
    AkshareAdapter(),
    TushareAdapter(),
    MootdxAdapter(),
    AmazingDataAdapter(),
]

# -------------------------
# Public API
# -------------------------

def read_daily_history(
    codes: CodesLike,
    start_date: Optional[Union[str, date, datetime]] = None,
    end_date: Optional[Union[str, date, datetime]] = None,
    sources_priority: Optional[Sequence[str]] = None,
    adjust: str = "",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Read daily OHLCV(+amount) history with multi-source fallback and unified format.

    - If a single code is provided, return a DataFrame indexed by datetime ascending.
    - If multiple codes are provided, return dict[code] -> DataFrame.

    Parameters:
    - codes: str or iterable of codes (e.g., '000001.SZ')
    - start_date, end_date: accepts 'YYYYMMDD', 'YYYY-MM-DD', datetime/date
    - sources_priority: optional explicit order of sources to try first (e.g., [DataSource.XTDATA, DataSource.AKSHARE])
    - adjust: per-source adjust option. For akshare: 'qfq'/'hfq'/''; for xtdata: maps to 'front'/'back'/'none'; for tushare: handled upstream.
    """
    code_list = _ensure_list(codes)
    _start = _normalize_date(start_date)
    _end = _normalize_date(end_date)
    options = ReadOptions(adjust=adjust or "")

    # Build adapter order: user-priority first, then remaining defaults
    adapter_map = {a.source: a for a in DEFAULT_ADAPTERS}
    ordered: List[BaseAdapter] = []
    if sources_priority:
        for s in sources_priority:
            if s in adapter_map:
                ordered.append(adapter_map[s])
    for a in DEFAULT_ADAPTERS:
        if a not in ordered:
            ordered.append(a)

    def fetch_one(code: str) -> Optional[pd.DataFrame]:
        last_error = None
        for adapter in ordered:
            try:
                df = adapter.fetch_daily_history(code, _start, _end, options)
                if df is not None and len(df) > 0:
                    logging.info(f"read_daily_history: {adapter.source} succeeded for {code}")
                    return df
            except Exception as e:
                last_error = e
                logging.debug(f"read_daily_history: {adapter.source} raised {e}")
        if last_error:
            logging.warning(f"All sources failed for {code}: {last_error}")
        else:
            logging.warning(f"All sources returned empty for {code}")
        return None

    if len(code_list) == 1:
        df = fetch_one(code_list[0])
        return df if df is not None else pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])  # empty

    # Multiple codes -> try batch first, then fallback to individual
    def fetch_batch() -> Optional[Dict[str, pd.DataFrame]]:
        """Try to fetch all codes using batch-capable adapters."""
        for adapter in ordered:
            if adapter.supports_batch():
                try:
                    batch_result = adapter.fetch_daily_history_batch(code_list, _start, _end, options)
                    if batch_result is not None:
                        # Return batch results directly
                        logging.info(f"read_daily_history: {adapter.source} batch succeeded for {len(code_list)} codes")
                        return batch_result

                except Exception as e:
                    logging.debug(f"read_daily_history: {adapter.source} batch failed: {e}")
                    continue
        return None

    # Try batch fetch first
    batch_result = fetch_batch()
    if batch_result is not None:
        return batch_result

    # Fallback to individual fetch
    result: Dict[str, pd.DataFrame] = {}
    for c in code_list:
        df = fetch_one(c)
        if df is not None and len(df) > 0:
            result[c] = df
        else:
            result[c] = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "amount"])  # empty
    return result


__all__ = [
    "DataCategory",
    "DataSource",
    "ReadOptions",
    "read_daily_history"
]

