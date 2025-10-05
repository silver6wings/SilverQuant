import time


ts_token_index = 0


def get_tushare_pro(debugging=False):
    import tushare as ts
    from credentials import TUSHARE_TOKEN
    for i in range(3):
        try:
            global ts_token_index
            ts_token_index += 1
            ts_token_index = ts_token_index % len(TUSHARE_TOKEN)
            ts_account = TUSHARE_TOKEN[ts_token_index]
            if debugging:
                print(ts_account)
            ts.set_token(ts_account[0])
            ts_pro_api = ts.pro_api()
            return ts_pro_api
        except:
            time.sleep(1)
    return None


if __name__ == '__main__':
    for i in range(10):
        pro = get_tushare_pro(debugging=True)
        df = pro.daily(ts_code="000001.SZ", start_date='20230606', end_date='20230610')
        print(df)
