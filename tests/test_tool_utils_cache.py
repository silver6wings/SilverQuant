import datetime

from tools.utils_cache import get_next_trading_date


def test_get_next_trading_date():
    dt = get_next_trading_date(datetime.datetime(2025, 9 , 29), 1)
    # print(dt)
    assert dt == '20250930'

    dt = get_next_trading_date(datetime.datetime(2025, 9 , 29), 2)
    # print(dt)
    assert dt == '20251009'
