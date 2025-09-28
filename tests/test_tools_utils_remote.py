from tools.utils_remote import get_mootdx_daily_history, ExitRight


def test_get_mootdx_daily_history():
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    code = '603929.SH'
    start_date = '20250915'
    df1 = get_mootdx_daily_history(code, start_date, '20250919', default_columns, ExitRight.QFQ)
    df2 = get_mootdx_daily_history(code, start_date, '20250922', default_columns, ExitRight.QFQ)
    # print(df1)
    # print(df2)
    df1_test = df1[df1['datetime'] == 20250919]
    df2_test = df2[df2['datetime'] == 20250919]

    assert len(df1) > 0 and len(df2) > 0
    assert df1_test['open'].values[0] - df2_test['open'].values[0] < 0.01
    assert df1_test['close'].values[0] - df2_test['close'].values[0] < 0.01
    assert df1_test['high'].values[0] - df2_test['high'].values[0] < 0.01
    assert df1_test['low'].values[0] - df2_test['low'].values[0] < 0.01
