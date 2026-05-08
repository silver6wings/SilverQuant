# -*- coding: utf-8 -*-
import datetime

import pytest

from tools import utils_cache
from tools.utils_cache import (
    get_next_trading_date,
    get_next_trading_date_str,
    get_prev_trading_date_list,
    get_trading_date_list,
)


def _clear_trade_day_caches():
    for fn in (
        utils_cache.get_next_trading_date_str,
        utils_cache.get_prev_trading_date_str,
        utils_cache.get_prev_trading_date_list,
        utils_cache.get_trading_date_list,
    ):
        fn.cache_clear()
    utils_cache.trade_day_cache.clear()


@pytest.fixture(autouse=True)
def _clear_trade_day_caches_around_tests():
    _clear_trade_day_caches()
    yield
    _clear_trade_day_caches()


@pytest.mark.parametrize(
    "anchor, offset, basic, expected",
    [
        (datetime.datetime(2025, 9, 29), 1, True, "20250930"),
        (datetime.datetime(2025, 9, 29), 2, True, "20251009"),
        (datetime.datetime(2025, 9, 29), 1, False, "2025-09-30"),
    ],
)
def test_get_next_trading_date(anchor, offset, basic, expected):
    assert get_next_trading_date(anchor, offset, basic_format=basic) == expected


@pytest.mark.parametrize(
    "today_str, offset, basic, expected",
    [
        ("2025-09-29", 1, True, "20250930"),
        ("20250929", 1, True, "20250930"),
        ("20250929", 1, False, "2025-09-30"),
    ],
)
def test_get_next_trading_date_str_compact_and_hyphen(today_str, offset, basic, expected):
    assert get_next_trading_date_str(today_str, offset, basic_format=basic) == expected


def test_get_trading_date_list_range_and_compact_equivalence():
    hyphen = get_trading_date_list("2025-10-12", "2025-10-22")
    compact = get_trading_date_list("20251012", "20251022")
    assert list(hyphen) == list(compact)
    assert len(hyphen) == 8
    assert hyphen[0] == "2025-10-13"
    assert hyphen[-1] == "2025-10-22"

    assert list(get_trading_date_list("20251013", "20251022")) == list(
        get_trading_date_list("2025-10-13", "2025-10-22")
    )


def test_get_trading_date_list_same_day_returns_input_shape():
    assert get_trading_date_list("2025-10-13", "2025-10-13") == ["2025-10-13"]
    assert get_trading_date_list("20251013", "20251013") == ["20251013"]


def test_get_trading_date_list_reversed_range_single_anchor():
    out = get_trading_date_list("2025-10-22", "2025-10-13")
    assert len(out) == 1
    assert out[0] == "2025-10-22"


def test_get_prev_trading_date_list_hyphen_vs_compact():
    a = get_prev_trading_date_list("2025-10-22", 8)
    b = get_prev_trading_date_list("20251022", 8)
    assert list(a) == list(b)
    assert len(a) == 8
    assert a[0] == "2025-10-10"
    assert a[-1] == "2025-10-21"
