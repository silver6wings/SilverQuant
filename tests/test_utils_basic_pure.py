# -*- coding: utf-8 -*-
import datetime

import pytest

from tools import utils_basic


@pytest.mark.parametrize(
    "sym, code",
    [
        ("000001", "000001.SZ"),
        ("600000", "600000.SH"),
        ("430001", "430001.BJ"),
        ("159915", "159915.SZ"),
        ("110001", "110001.SH"),
        (600000, "600000.SH"),
    ],
)
def test_symbol_to_code(sym, code):
    assert utils_basic.symbol_to_code(sym) == code


def test_symbol_to_code_unknown_prefix():
    assert utils_basic.symbol_to_code("999999") == "999999.--"


@pytest.mark.parametrize(
    "code, sym",
    [
        ("000001.SZ", "000001"),
        ("600000.SH", "600000"),
    ],
)
def test_code_to_symbol(code, sym):
    assert utils_basic.code_to_symbol(code) == sym


def test_code_to_symbol_invalid():
    with pytest.raises(AssertionError):
        utils_basic.code_to_symbol("600000")


@pytest.mark.parametrize(
    "code, sina",
    [
        ("000001.SZ", "sz000001"),
        ("600000.SH", "sh600000"),
        ("430001.BJ", "bj430001"),
    ],
)
def test_code_sina_roundtrip(code, sina):
    assert utils_basic.code_to_sina_symbol(code) == sina
    assert utils_basic.sina_symbol_to_code(sina) == code


@pytest.mark.parametrize(
    "raw, code",
    [
        ("sh600000", "600000.SH"),
        ("SH600000", "600000.SH"),
        ("not8ch", "not8ch"),
    ],
)
def test_sina_symbol_to_code_edge(raw, code):
    assert utils_basic.sina_symbol_to_code(raw) == code


@pytest.mark.parametrize(
    "code, tdx",
    [
        ("000001.SZ", "0000001"),
        ("600000.SH", "1600000"),
        ("430001.BJ", "2430001"),
    ],
)
def test_code_tdxsymbol_roundtrip(code, tdx):
    assert utils_basic.code_to_tdxsymbol(code) == tdx
    assert utils_basic.tdxsymbol_to_code(tdx) == code


@pytest.mark.parametrize(
    "sym, tdx",
    [
        ("600000", "1600000"),
        ("000001", "0000001"),
        ("430001", "2430001"),
    ],
)
def test_symbol_to_tdxsymbol(sym, tdx):
    assert utils_basic.symbol_to_tdxsymbol(sym) == tdx


@pytest.mark.parametrize(
    "sym, gm",
    [
        ("000001", "SZSE.000001"),
        ("600000", "SHSE.600000"),
        ("430001", "BJSE.430001"),
    ],
)
def test_gm_symbol_roundtrip(sym, gm):
    assert utils_basic.symbol_to_gmsymbol(sym) == gm
    assert utils_basic.gmsymbol_to_symbol(gm) == sym


def test_code_gmsymbol_chain():
    assert utils_basic.code_to_gmsymbol("600000.SH") == "SHSE.600000"
    assert utils_basic.gmsymbol_to_code("SHSE.600000") == "600000.SH"


@pytest.mark.parametrize(
    "s, fn, exp",
    [
        ("000001", utils_basic.is_symbol, True),
        ("110001", utils_basic.is_symbol, True),
        ("600000", utils_basic.is_symbol, True),
        ("999999", utils_basic.is_symbol, False),
        ("600000", utils_basic.is_stock, True),
        ("159915", utils_basic.is_stock, False),
        ("000001.SH", utils_basic.is_stock_code, False),
        ("000001.SZ", utils_basic.is_stock_code, True),
        ("600000", utils_basic.is_stock_10cm, True),
        ("300001", utils_basic.is_stock_20cm, True),
        ("430001", utils_basic.is_stock_30cm, True),
        ("300001", utils_basic.is_stock_cy, True),
        ("688001", utils_basic.is_stock_kc, True),
        ("430001", utils_basic.is_stock_bj, True),
        ("159915", utils_basic.is_fund_etf, True),
        ("110001", utils_basic.is_bond, True),
    ],
)
def test_code_kind_predicates(s, fn, exp):
    assert fn(s) is exp


@pytest.mark.parametrize(
    "sym, ex",
    [
        ("000001", "SZ"),
        ("600000", "SH"),
        ("430001", "BJ"),
        ("999999", ""),
    ],
)
def test_get_symbol_exchange(sym, ex):
    assert utils_basic.get_symbol_exchange(sym) == ex


def test_get_code_exchange():
    assert utils_basic.get_code_exchange("600000.SH") == "SH"


@pytest.mark.parametrize(
    "n, ch",
    [
        (0, "0"),
        (500, "5"),
        (1000, "a"),
        (3600, "A"),
        (6200, "."),
    ],
)
def test_map_num_to_chr(n, ch):
    assert utils_basic.map_num_to_chr(n) == ch


@pytest.mark.parametrize(
    "t, inside",
    [
        ("09:30", True),
        ("11:30", True),
        ("11:31", False),
        ("13:00", True),
        ("14:57", True),
        ("14:58", False),
        ("08:00", False),
    ],
)
def test_is_in_continuous_auction(t, inside):
    assert utils_basic.is_in_continuous_auction(t) is inside


@pytest.mark.parametrize(
    "t, pct",
    [
        ("09:30:00", 0.0),
        ("10:00:00", pytest.approx(0.125, rel=1e-9)),
        ("13:00:00", pytest.approx(0.5, rel=1e-9)),
        ("08:00:00", -1),
    ],
)
def test_get_current_time_percentage(t, pct):
    assert utils_basic.get_current_time_percentage(t) == pct


@pytest.mark.parametrize(
    "sym, up, down",
    [
        ("600000", 1.1, 0.9),
        ("300001", 1.2, 0.8),
        ("430001", 1.3, 0.7),
        ("830001", 1.3, 0.7),
    ],
)
def test_limit_rates(sym, up, down):
    assert utils_basic.get_limiting_up_rate(sym) == up
    assert utils_basic.get_limiting_down_rate(sym) == down


@pytest.mark.parametrize(
    "sym, pre, lim_up, lim_dn",
    [
        ("600000", 10.0, 11.0, 9.0),
        ("300001", 10.0, 12.0, 8.0),
        ("600000", 0.0, 0.0, 0.0),
        ("600000", -1.0, 0.0, 0.0),
    ],
)
def test_limit_prices(sym, pre, lim_up, lim_dn):
    assert utils_basic.get_limit_up_price(sym, pre) == lim_up
    assert utils_basic.get_limit_down_price(sym, pre) == lim_dn


def test_time_diff_seconds():
    a = datetime.time(10, 0, 0)
    b = datetime.time(9, 0, 0)
    assert utils_basic.time_diff_seconds(a, b) == 3600


@pytest.mark.parametrize(
    "h, m, s, past",
    [
        (9, 0, 0, 0),
        (9, 30, 0, 0),
        (10, 0, 0, 1800),
        (11, 30, 0, 7200),
        (13, 0, 0, 7200),
        (14, 0, 0, 10800),
        (15, 0, 0, 14400),
    ],
)
def test_hms_to_past_seconds(h, m, s, past):
    assert utils_basic.hms_to_past_seconds(h, m, s) == past


def test_xt_time_tag_to_hms_and_past():
    assert utils_basic.xt_time_tag_to_hms("20250102 14:00:05") == (14, 0, 5)
    assert utils_basic.xt_time_tag_to_past_seconds("20250102 14:00:05") == utils_basic.hms_to_past_seconds(
        14, 0, 5
    )


def test_xt_time_to_hms_short_timestamp():
    assert utils_basic.xt_time_to_hms(1234567890) == (0, 0, 0)
