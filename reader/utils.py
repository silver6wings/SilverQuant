"""Utility functions for the multi-source reader."""

from __future__ import annotations

import logging
import pandas as pd
from datetime import date, datetime, time, timedelta
from typing import Dict, Iterable, List, Optional, Tuple, Union

# Type aliases
CodeLike = Union[str, int]
CodesLike = Union[CodeLike, Iterable[CodeLike]]

# Trading sessions for China A-shares
TRADING_SESSIONS = [
    (time(9, 30, 0), time(11, 30, 0)),
    (time(13, 0, 0), time(15, 0, 0)),
]


def _is_trading_day(d: date) -> bool:
    """Check if a date is a trading day (weekdays Mon-Fri)."""
    # Weekdays Mon-Fri as trading days; holidays not accounted for
    return d.weekday() < 5

def _ensure_list(codes: CodesLike) -> List[str]:
    """Ensure codes is a list of strings."""
    if codes is None:
        return []
    if isinstance(codes, (list, tuple, set)):
        return [str(c) for c in codes]
    return [str(codes)]


def _normalize_date(d: Optional[Union[str, date, datetime]]) -> Optional[str]:
    """Return YYYYMMDD string or None."""
    if d is None:
        return None
    if isinstance(d, datetime):
        return d.strftime("%Y%m%d")
    if isinstance(d, date):
        return d.strftime("%Y%m%d")
    s = str(d).strip()
    # Accept 'YYYY-MM-DD' or 'YYYY/MM/DD' or 'YYYYMMDD'
    if len(s) == 8 and s.isdigit():
        return s
    try:
        return datetime.strptime(s.replace("/", "-"), "%Y-%m-%d").strftime("%Y%m%d")
    except Exception:
        logging.warning(f"Unrecognized date format: {d}")
        return s






