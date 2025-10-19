import datetime
import pytest
from tools.utils_cache import get_next_trading_date, get_prev_trading_date_list, get_trading_date_list


def test_get_next_trading_date():
    dt = get_next_trading_date(datetime.datetime(2025, 9 , 29), 1)
    # print(dt)
    assert dt == '20250930'

    dt = get_next_trading_date(datetime.datetime(2025, 9 , 29), 2)
    # print(dt)
    assert dt == '20251009'

def test_get_trading_date_list():
    date_list = get_trading_date_list('20251012', '20251022')
    assert len(date_list) == 8
    assert date_list[0] == '2025-10-13'
    assert date_list[-1] == '2025-10-22'
    
def test_get_prev_trading_date_list():
    date_list = get_prev_trading_date_list('2025-10-22', 8)
    assert len(date_list) == 8
    assert date_list[0] == '2025-10-10'
    assert date_list[-1] == '2025-10-21'
