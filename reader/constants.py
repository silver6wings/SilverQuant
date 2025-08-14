"""Constants and enumerations for the multi-source reader."""

from dataclasses import dataclass


class DataCategory:
    """Supported data categories for the multi-source reader.

    Start with the most common demand: daily OHLCV(+amount) history.
    """

    DAILY_HISTORY = "daily_history"


class DataSource:
    """Data source enumeration (aligned with reader.joinquant_compat)."""

    XTDATA = "xtdata"
    AKSHARE = "akshare"
    MOOTDX = "mootdx"
    AMAZING_DATA = "amazing_data"
    TUSHARE = "tushare"


@dataclass
class ReadOptions:
    """Options for reading data from various sources."""
    adjust: str = ""  # "qfq"/"hfq"/"" etc., interpreted per source
    # Future extension: frequency, fields, fill options ...
