"""
广为流传的六脉神剑，当做例子其实也不咋赚钱

DIFF:=EMA(CLOSE,8)-EMA(CLOSE,13);
DEA:=EMA(DIFF,5);

//DRAWICON(DIFF>DEA,1,1);
//DRAWICON(DIFF<DEA,1,2);
//DRAWTEXT(ISLASTBAR=1,1,'. MACD'),COLORFFFFFF;
SL1 := DIFF>DEA;

RSV:= (CLOSE-LLV(LOW,8))/(HHV(HIGH,8)-LLV(LOW,8))*100;
K:=SMA(RSV,3,1);
D:=SMA(K,3,1);

//DRAWICON(K>D,2,1);
//DRAWICON(K<D,2,2);
//DRAWTEXT(ISLASTBAR=1,2,'. KDJ'),COLORFFFFFF;
SL2:=K>D;

RSV:=REF(CLOSE,1);
RSI1:=(SMA(MAX(CLOSE-RSV,0),5,1))/(SMA(ABS(CLOSE-RSV),5,1))*100;
RSI2:=(SMA(MAX(CLOSE-RSV,0),13,1))/(SMA(ABS(CLOSE-RSV),13,1))*100;

//DRAWICON(RSI1>RSI2,3,1);
//DRAWICON(RSI1<RSI2,3,2);
//DRAWTEXT(ISLASTBAR=1,3,'. RSI'),COLORFFFFFF;
SL3:=RSI1>RSI2;

ZN:=-(HHV(HIGH,13)-CLOSE)/(HHV(HIGH,13)-LLV(LOW,13))*100;
JC1:=SMA(ZN,3,1);
JC2:=SMA(JC1,3,1);

//DRAWICON(JC1>JC2,4,1);
//DRAWICON(JC1<JC2,4,2);
//DRAWTEXT(ISLASTBAR=1,4,'. JC'),COLORFFFFFF;
SL4:=JC1>JC2;

BBI:=(MA(CLOSE,3)+MA(CLOSE,5)+MA(CLOSE,8)+MA(CLOSE,13))/4;

//DRAWICON(CLOSE>BBI,5,1);
//DRAWICON(CLOSE<BBI,5,2);
//DRAWTEXT(ISLASTBAR=1,5,'. BBI'),COLORFFFFFF;
SL5:=CLOSE>BBI;

MTM:=CLOSE-REF(CLOSE,1);
MMS:=100*EMA(EMA(MTM,5),3)/EMA(EMA(ABS(MTM),5),3);
MMM:=100*EMA(EMA(MTM,13),8)/EMA(EMA(ABS(MTM),13),8);

//DRAWICON(MMS>MMM,6,1);
//DRAWICON(MMS<MMM,6,2);
//DRAWTEXT(ISLASTBAR=1,6,'. ZLMM'),COLORFFFFFF;
SL6:=MMS>MMM;

GZ:= SL1 AND SL2 AND SL3 AND SL4 AND SL5 AND SL6;

LM:= 5;
PC:= REF(COUNT(GZ = 0, LM), 1);
LMZJ:= IF(GZ = 1 AND (REF(GZ, 1) = 0) AND (PC = LM), 6, 0),NODRAW;

DD:= C > O;                              // 当日阳线
EE:= C > REF(C, 1) * 1.02;               // 当日涨幅 > 2%
FF:= (H - MAX(C, O)) / (H - L) < 0.2;    // 上影线 < K线长度 * 0.2

BUY: LMZJ AND DD AND EE AND FF;
HOLD: IF(GZ, 6, 0), NODRAW;
SELL: IF(GZ = 0 AND REF(GZ, 1) = 1, 6, 0);

DRAWICON(DIFF>DEA,1,4);
DRAWICON(DIFF<DEA,1,5);

DRAWICON(K>D,2,4);
DRAWICON(K<D,2,5);

DRAWICON(RSI1>RSI2,3,4);
DRAWICON(RSI1<RSI2,3,5);

DRAWICON(JC1>JC2,4,4);
DRAWICON(JC1<JC2,4,5);

DRAWICON(CLOSE>BBI,5,4);
DRAWICON(CLOSE<BBI,5,5);

DRAWICON(MMS>MMM,6,4);
DRAWICON(MMS<MMM,6,5);

DRAWICON(LMZJ,6.6,8);
DRAWICON(SELL,6.8,9);

STICKLINE(GZ,0,6,0.6,1),COLORMAGENTA;
STICKLINE(BUY,0,6,0.6,0),COLORYELLOW;
"""

import warnings
warnings.filterwarnings("ignore")

from mytt.MyTT_advance import *


def select(df: pd.DataFrame, code: str, quote: dict):
    O = df.open
    H = df.high
    L = df.low
    C = df.close
    # VOL = df.volume
    # AMOUNT = df.amount

    DIFF = EMA(C, 8) - EMA(C, 13)
    DEA = EMA(DIFF, 5)
    SL1 = DIFF > DEA

    RSV = (C - LLV(L, 8)) / (HHV(H, 8) - LLV(L, 8)) * 100
    K = SMA(RSV, 3, 1)
    D = SMA(K, 3, 1)
    SL2 = K > D

    QF = REF(C, 1)
    RSI1 = (SMA(MAX(C - QF, 0), 5, 1)) / (SMA(ABS(C - QF), 5, 1)) * 100
    RSI2 = (SMA(MAX(C - QF, 0), 13, 1)) / (SMA(ABS(C - QF), 13, 1)) * 100
    SL3 = RSI1 > RSI2

    ZN = -(HHV(H, 13) - C) / (HHV(H, 13) - LLV(L, 13)) * 100
    JC1 = SMA(ZN, 3, 1)
    JC2 = SMA(JC1, 3, 1)
    SL4 = JC1 > JC2

    BBI = (MA(C, 3) + MA(C, 6) + MA(C, 12) + MA(C, 24)) / 4
    SL5 = C > BBI

    MTM = C - REF(C, 1)
    MMS = 100 * EMA(EMA(MTM, 5), 3) / EMA(EMA(ABS(MTM), 5), 3)
    MMM = 100 * EMA(EMA(MTM, 13), 8) / EMA(EMA(ABS(MTM), 13), 8)
    SL6 = MMS > MMM

    GZ = SL1 & SL2 & SL3 & SL4 & SL5 & SL6

    LM = 5
    PC = REF(COUNT(~GZ, LM), 1)
    LMZJ = GZ & (REF(~GZ, 1)) & (PC == LM)

    df['SL1'] = SL1
    df['SL2'] = SL2
    df['SL3'] = SL3
    df['SL4'] = SL4
    df['SL5'] = SL5
    df['SL6'] = SL6
    df['GZ'] = GZ
    df['PC'] = PC

    df['DD'] = C > O
    df['EE'] = C > REF(C, 1) * 1.02
    df['FF'] = (H - MAX(C, O)) / (H - L) < 0.2

    df['PASS'] = LMZJ & df['DD'] & df['EE'] & df['FF']

    return df
