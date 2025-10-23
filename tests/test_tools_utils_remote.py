import pytest

from tools.utils_cache import get_prev_trading_date_str
from tools.utils_mootdx import *
from tools.utils_remote import ExitRight, get_mootdx_daily_history


@pytest.mark.local_only
def test_get_mootdx_daily_history():
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    code = '603929.SH' #20250922 每股除权除息 10派10
    start_date = '20250915'
    df1 = get_mootdx_daily_history(code, start_date, '20250919', default_columns, ExitRight.QFQ)
    df2 = get_mootdx_daily_history(code, start_date, '20250922', default_columns, ExitRight.QFQ)
    # print(df1)
    # print(df2)
    df1_test = df1[df1['datetime'] == 20250919]
    df2_test = df2[df2['datetime'] == 20250919]
    #20250922 每股除权除息 10派10 故0919日期open需大于0922日期
    assert len(df1) > 0 and len(df2) > 0
    assert df1_test['open'].values[0] - 41.69 < 0.01
    assert df1_test['open'].values[0] - df2_test['open'].values[0] > 0.01
    assert df1_test['close'].values[0] - df2_test['close'].values[0] > 0.01
    assert df1_test['high'].values[0] - df2_test['high'].values[0] > 0.01
    assert df1_test['low'].values[0] - df2_test['low'].values[0] > 0.01


@pytest.mark.local_only
def test_get_tdxzip_history():
    buffer = download_tdx_hsjday()
    assert buffer != False
    buffer.seek(0, io.SEEK_END)
    response_length = buffer.tell()
    buffer.close()
    file_size = response_length / 1024 / 1024
    assert file_size > 480.0
    
    full_history = get_tdxzip_history(adjust = ExitRight.QFQ)
    code = '002594.SZ' # 比亚迪
    xdxr_date = 20250729 # 比亚迪这次除权差异较大
    xrxr_date_close = 111.42
    assert len(full_history) > 5430
    assert len(full_history[code]) == 550
    assert full_history[code].loc[full_history[code]['datetime'] == xdxr_date]['close'].values[0] - xrxr_date_close < 0.01
    # '603929.SH' 20250919 open价格应该是 40.69
    assert full_history['603929.SH'].loc[full_history['603929.SH']['datetime'] == 20250919]['close'].values[0] - 40.69 < 0.01
    
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    day_count = 120
    start = get_prev_trading_date_list(curr_date, day_count)[0].replace('-','')
    end  = get_prev_trading_date_str(curr_date, 1)
    days = len(get_trading_date_list(start, end))
    assert days == day_count
    
    target_codes = ['000555.SZ', '002017.SZ', '603367.SH', '600078.SH','688098.SH','688001.SH', '920225.BJ','920022.BJ']
    default_columns: list[str] = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    cache_history = {}
    for code in target_codes:
        if code in full_history:
            cache_history[code] = full_history[code][default_columns].tail(days).copy()
    assert len(cache_history) == len(target_codes)
    assert len(cache_history['000555.SZ']) == day_count


# 远程Github服务器访问接口会有问题，也只跑本地测试即可
@pytest.mark.local_only
def test_check_xdxr_cache():
    if os.path.isfile(PATH_TDX_XDXR):
        cache_xdxr_orig = load_pickle(PATH_TDX_XDXR)
        updatedtime = cache_xdxr_orig.get('updatedtime', None)
        updated_date = updatedtime.strftime('%Y-%m-%d') # 文件存在的话
        assert updatedtime is not None
    else:
        cache_xdxr_orig = {}
        updated_date = get_prev_trading_date_str(datetime.datetime.now().strftime('%Y-%m-%d'), 0)

    xcdf, divicount = get_dividend_code_from_baidu(updated_date)
    assert len(xcdf) > 0 # 百度除权信息接口正常
    assert len(xcdf) == divicount # 百度除权信息接口正常

    xdxr_count = len(cache_xdxr_orig)
    test_code = symbol_to_code(xcdf.iloc[-1]['code'])
    print(test_code)
    if xdxr_count > 0:
        for row in xcdf.itertuples():
            symbol = row.code
            code = symbol_to_code(symbol)
            cache_xdxr_orig.pop(code)
            test_code = code
        assert xdxr_count > len(cache_xdxr_orig)
        save_pickle(PATH_TDX_XDXR, cache_xdxr_orig)
        
    check_xdxr_cache(adjust=ExitRight.QFQ, force_refresh_updated_date=True)
    cache_xdxr_new = load_pickle(PATH_TDX_XDXR)
    assert len(cache_xdxr_new) > 0
    if xdxr_count > 0:
        assert len(cache_xdxr_new) == xdxr_count
    assert len(cache_xdxr_new[test_code]) > 0
