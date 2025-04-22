"""
打板的样例公式

AA: C >= REF(C, 1) * LIMITINGUPRATE;
BB: C > REF(HHV(C, 30), 1);
CC: C < LLV(C, 60) * 2.0;
DD: VOL < REF(VOL, 1) * 3.00;

MA10 := MA(C, 10);
MA20 := MA(C, 20);
MA30 := MA(C, 30)
MA60 := MA(C, 60);
EE: H < MA60 * 1.3;;
FF: (SLOPE(MA10, 3) > 0) AND (SLOPE(MA20, 3) > 0) AND (SLOPE(MA30, 3) > 0) AND (SLOPE(MA60, 3) > 0);
GG: (C > MA10) AND (MA10 > MA20) AND (MA20 > MA30) AND (MA30 > MA60);
HH: COUNT(C >= REF(C, 1) * LIMITINGUPRATE, 60) < 3;
II: COUNT(C >= REF(C, 1) * LIMITINGUPRATE, 3) < 1;
JJ: ASKVOL(1) <= 0;

BUY: AA AND BB AND CC AND DD AND EE AND FF AND GG AND HH AND II AND JJ;
"""

from mytt.MyTT import *
from mytt.MyTT_advance import *
from tools.utils_basic import get_limiting_up_rate


def select(df: pd.DataFrame, code: str, quote: dict):
    LIMITINGUPRATE = get_limiting_up_rate(code)

    # O = df.open
    H = df.high
    # L = df.low
    C = df.close
    VOL = df.volume
    # AMOUNT = df.amount

    df['AA'] = C >= REF(C, 1) * LIMITINGUPRATE  # 价格是涨停
    df['BB'] = C > REF(HHV(C, 30), 1)           # 当日首次突破前30日内收盘新高
    df['CC'] = C < LLV(C, 60) * 2.0             # 现价 < 前60日最低 * 2
    df['DD'] = VOL < REF(VOL, 1) * 3.00         # 当日成交量 < 昨日成交 * 4.00

    MA10 = MA(C, 10)
    MA20 = MA(C, 20)
    MA30 = MA(C, 30)
    MA60 = MA(C, 60)
    df['EE'] = H < MA60 * 1.3                                   # 当日最高价 < 60日均线价格 * 1.3
    df['FF'] = (SLOPE(MA10, 3) > 0) & (SLOPE(MA20, 3) > 0) & \
               (SLOPE(MA30, 3) > 0) & (SLOPE(MA60, 3) > 0)      # 均线上升趋势
    df['GG'] = (C > MA10) & (MA10 > MA20) & \
               (MA20 > MA30) & (MA30 > MA60)                    # 现价10日20日30日60日均线呈多头排列
    df['HH'] = COUNT(C >= REF(C, 1) * LIMITINGUPRATE, 60) < 3   # 60天内涨停次数小于三次
    df['II'] = COUNT(C >= REF(C, 1) * LIMITINGUPRATE, 3) < 1    # 最近三天没有过涨停

    df['PASS'] = df['AA']
    for i in range(ord('B'), ord('I') + 1):
        df['PASS'] = df['PASS'] & df[f'{chr(i)}{chr(i)}']

    return df
