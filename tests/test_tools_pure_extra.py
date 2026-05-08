# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pandas as pd
import pytest

from mytt.MyTT import RD, RET
from tools import utils_cache
from tools.constants import DataSource, ExitRight, IndexSymbol, InfoItem


def test_constants_str_enum_values():
    assert DataSource.AKSHARE == "akshare"
    assert ExitRight.QFQ == "qfq"
    assert IndexSymbol.INDEX_HS_300 == "000300"
    assert InfoItem.DayCount == "day_count"
    assert str(DataSource.MOOTDX) == "mootdx"


def test_index_constituent_df_ok():
    assert utils_cache._index_constituent_df_ok(None) is False
    assert utils_cache._index_constituent_df_ok(pd.DataFrame()) is False
    assert utils_cache._index_constituent_df_ok(pd.DataFrame({"品种代码": [1]})) is True
    assert utils_cache._index_constituent_df_ok(pd.DataFrame({"成分券代码": [1]})) is True
    assert utils_cache._index_constituent_df_ok(pd.DataFrame({"x": [1]})) is False


def test_filter_none_st_out():
    df = pd.DataFrame(
        {
            "code": ["1", "2", "3", "4"],
            "name": ["正常", "*ST测试", "退市股", "退市整理"],
        }
    )
    out = utils_cache._filter_none_st_out(df)
    assert list(out["name"]) == ["正常"]


@pytest.mark.parametrize(
    "when, report",
    [
        (datetime.datetime(2026, 3, 1), "20251231"),
        (datetime.datetime(2026, 8, 1), "20260630"),
        (datetime.datetime(2026, 10, 1), "20260930"),
    ],
)
def test_get_recent_fhps_report_date(when, report):
    assert utils_cache._get_recent_fhps_report_date(when) == report


def test_mytt_rd_ret():
    assert RD(1.23456, 2) == 1.23
    assert RET([1, 2, 3]) == 3
    assert RET(np.array([1.0, 2.0]), 2) == 1.0
