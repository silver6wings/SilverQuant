import numpy as np
import pandas as pd
from typing import Union, Any


# ============
#  单行单列操作
# ============

# 获取最后一个数值
def LAST_(n: pd.Series) -> Any:
    return n.iloc[-1]


# 最后一个布尔数值转换成整数01
def INT_(n: Union[bool, pd.Series]) -> int:
    t = n.iloc[-1] if type(n) == pd.Series else n
    return 1 if t else 0


# 最后一个布尔数值转换成整数10
def NOT_(n: Union[bool, pd.Series]) -> int:
    t = n.iloc[-1] if type(n) == pd.Series else n
    return 0 if t else 1


# 获取一个整数
def INTPART_(n: Union[float, pd.Series]) -> int:
    t = n.iloc[-1] if type(n) == pd.Series else n
    return int(t)


# ============
#  单行多列操作
# ============


# 比较两列最后一个数值取较大值
def MAX_(a: Union[int, float, pd.Series], b: Union[int, float, pd.Series]):
    x = a.iloc[-1] if type(a) == pd.Series else a
    y = b.iloc[-1] if type(b) == pd.Series else b
    return max(x, y)


# 比较两列最后一个数值取较小值
def MIN_(a: Union[int, float, pd.Series], b: Union[int, float, pd.Series]):
    x = a.iloc[-1] if type(a) == pd.Series else a
    y = b.iloc[-1] if type(b) == pd.Series else b
    return min(x, y)


# 条件判断取值
def IF_(e: Union[int, pd.Series], n1: Union[int, pd.Series], n2: Union[int, pd.Series]):
    te = e.iloc[-1] if type(e) == pd.Series else e
    t1 = n1.iloc[-1] if type(n1) == pd.Series else n1
    t2 = n2.iloc[-1] if type(n2) == pd.Series else n2
    return t1 if te == 1 else t2


# ============
#  预知多行操作
# ============


# 向前数p周期内，e==1的个数
def COUNT_(e: pd.Series, p: Union[int, pd.Series]) -> int:
    # e需要是01，从当天为今天，向前数p-1天
    t = p.iloc[-1] if type(p) == pd.Series else p
    arr = e.iloc[-int(t):]
    return int(np.sum(arr))


# n1上穿n2
def CROSS_(n1: pd.Series, n2: Union[int, pd.Series]) -> int:
    t = n2.iloc[-1] if type(n2) == pd.Series else n2
    return 1 if n1.iloc[-1] > t > n1.iloc[-2] else 0


# 向前数p个数字
def REF_(n: pd.Series, p: Union[int, pd.Series]):
    t = p.iloc[-1] if type(p) == pd.Series else p
    assert t >= 0, "参数应该>=0"
    return n.iloc[-1 - t]


# 从当天按1算
def HHV_(n: pd.Series, p: Union[int, pd.Series]):
    t = p.iloc[-1] if type(p) == pd.Series else p
    t = 1 if t == 0 else t
    return np.max(n.iloc[-int(t):])


# 从当天按1算
def LLV_(n: pd.Series, p: Union[int, pd.Series]):
    t = p.iloc[-1] if type(p) == pd.Series else p
    t = 1 if t == 0 else t
    return np.min(n.iloc[-int(t):])


# 均线，至少一天
def MA_(n: pd.Series, p: Union[int, pd.Series]):
    t = p.iloc[-1] if type(p) == pd.Series else p
    assert t > 0, "参数应该>0"
    return np.sum(n.iloc[-int(t):]) / t


# ============
#  未知多行操作
# ============


# 上一个e==1距离当前的周期数值
def BARSLAST_(e: pd.Series):
    t = 0
    while t < len(e):
        if e.iloc[-1 - t] == 1:
            return t
        t += 1
    return t  # 找不到数据则为列长度


# 上一个e==1时对应当时n的数值
def VALUEWHEN_(e: pd.Series, n: pd.Series):
    t = 0
    while t < len(e):
        if e.iloc[-1 - t] == 1:
            return n.iloc[-1 - t]
        t += 1
    return n.iloc[0]  # 找不到数据则为第0个数据
