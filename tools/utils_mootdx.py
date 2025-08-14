import datetime
from mootdx.quotes import Quotes


mootdx_client = Quotes.factory(market='std')


def get_quotes(code_list: list[str]):
    if code_list is None or len(code_list) == 0:
        return {}

    symbol_list = [code.split('.')[0] for code in code_list]

    df = mootdx_client.quotes(symbol=symbol_list)

    result = {}
    for _, row in df.iterrows():
        # 构建股票代码（考虑market字段：0为深交所，1为上交所, 2为北交所）
        market_suffix = '.SZ' if row['market'] == 0 else ('.SH' if row['market'] == 1 else '.BJ')
        stock_code = f"{row['code']}{market_suffix}"

        time_str = row['servertime']    # 转换servertime为毫秒时间戳
        date_str = datetime.datetime.today().strftime('%Y-%m-%d')

        datetime_obj = datetime.datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S.%f')
        timestamp_ms = int(datetime_obj.timestamp() * 1000)

        ask_price = [row[f'ask{i + 1}'] for i in range(5)]
        bid_price = [row[f'bid{i + 1}'] for i in range(5)]
        ask_vol = [row[f'ask_vol{i + 1}'] for i in range(5)]
        bid_vol = [row[f'bid_vol{i + 1}'] for i in range(5)]

        stock_data = {
            'time': timestamp_ms,
            'lastPrice': row['price'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'lastClose': row['last_close'],
            'amount': row['amount'],
            'volume': row['vol'],
            'pvolume': row['vol'] * 100,  # 手转股
            # 'stockStatus': 0,
            # 'openInt': 0,
            # 'transactionNum': 0,
            # 'lastSettlementPrice': 0.0,
            # 'settlementPrice': 0.0,
            # 'pe': 0.0,
            'askPrice': ask_price,
            'bidPrice': bid_price,
            'askVol': ask_vol,
            'bidVol': bid_vol,
            # 'volRatio': 0.0,
            # 'speed1Min': 0.0,
            # 'speed5Min': 0.0
        }
        result[stock_code] = stock_data

    return result


if __name__ == '__main__':
    import json
    quotes = get_quotes(['000001.SZ', '600000.SH'])
    for code in quotes:
        print(json.dumps(quotes[code], indent=4))
