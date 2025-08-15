# SilverQuant 数据处理与存储模块深度分析

## 概述

SilverQuant项目采用了多层次、模块化的数据处理架构，实现了从数据获取、处理、缓存到应用的完整数据流管线。项目支持多种数据源，具备智能缓存机制，并提供了丰富的技术指标计算功能。

## 1. 数据架构总览

### 1.1 数据流向图
```
[外部数据源] → [数据获取层] → [数据处理层] → [缓存存储层] → [业务应用层]
    ↓              ↓             ↓             ↓             ↓
AKShare        utils_remote   MyTT技术指标   daily_history   交易策略
Tushare        utils_mootdx   数据清洗      utils_cache     选股算法
MooTDX         utils_basic    格式转换      磁盘缓存        风控系统
问财API        代码转换       数据验证      内存缓存        通知系统
```

### 1.2 核心模块关系
- **数据获取模块**: `tools/utils_remote.py`、`tools/utils_mootdx.py`
- **数据处理模块**: `mytt/MyTT.py`、`tools/utils_basic.py`
- **缓存管理模块**: `reader/daily_history.py`、`tools/utils_cache.py`
- **数据应用模块**: 各选股器、交易模块

## 2. 数据源接入层

### 2.1 多数据源支持架构

项目支持以下主要数据源：

#### AKShare数据源 (主力)
```python
# 文件: tools/utils_remote.py
def get_ak_daily_history(code, start_date, end_date, columns=None, adjust=''):
    import akshare as ak
    df = ak.stock_zh_a_hist(
        symbol=code_to_symbol(code),
        start_date=start_date,
        end_date=end_date,
        adjust=adjust,
        period='daily',
    )
    # 字段标准化
    df = df.rename(columns={
        '日期': 'datetime',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low', 
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
    })
```

#### Tushare数据源 (备用高频)
```python
# 文件: tools/utils_remote.py + reader/tushare_agent.py
def get_ts_daily_history(code, start_date, end_date, columns=None, adjust=''):
    from reader.tushare_agent import get_tushare_pro
    pro = get_tushare_pro()
    df = pro.daily(
        ts_code=code,
        start_date=start_date,
        end_date=end_date,
    )
    return ts_to_standard(df)  # 数据格式标准化
```

#### MooTDX本地行情源
```python
# 文件: tools/utils_mootdx.py
def get_quotes(code_list):
    symbol_list = [code.split('.')[0] for code in code_list]
    df = mootdx_client.quotes(symbol=symbol_list)
    # 实时行情数据结构化处理
    for _, row in df.iterrows():
        stock_data = {
            'time': timestamp_ms,
            'lastPrice': row['price'],
            'askPrice': [row[f'ask{i+1}'] for i in range(5)],
            'bidPrice': [row[f'bid{i+1}'] for i in range(5)],
            # ... 完整的五档行情数据
        }
```

### 2.2 数据源切换机制

```python
class DataSource:
    AKSHARE = 0
    TUSHARE = 1

def get_daily_history(code, start_date, end_date, columns=None, adjust='', data_source=DataSource.AKSHARE):
    if data_source == DataSource.TUSHARE:
        return get_ts_daily_history(code, start_date, end_date, columns, adjust)
    return get_ak_daily_history(code, start_date, end_date, columns, adjust)
```

### 2.3 问财AI数据获取

```python
# 文件: tools/utils_remote.py
def get_wencai_codes(queries):
    import pywencai
    result = set()
    for query in queries:
        df = pywencai.get(query=query, perpage=100, loop=True)
        if df is not None and df.shape[0] > 0:
            result.update(df['股票代码'].values)
    return list(result)
```

## 3. 数据处理层

### 3.1 代码格式转换系统

项目实现了多种股票代码格式的相互转换：

```python
# 文件: tools/utils_basic.py

# Symbol → Code (如: 000001 → 000001.SZ)
def symbol_to_code(symbol):
    if symbol[:2] in ['00', '30', '15', '12']:
        return f'{symbol}.SZ'
    elif symbol[:2] in ['60', '68', '51', '52', '53', '56', '58', '11']:
        return f'{symbol}.SH'
    elif symbol[:2] in ['83', '87', '43', '82', '88', '92']:
        return f'{symbol}.BJ'

# 通达信格式转换 (如: 000001.SZ → 0000001)
def code_to_tdxsymbol(code):
    [symbol, exchange] = code.split('.')
    if exchange == 'SZ': return '0' + symbol
    elif exchange == 'SH': return '1' + symbol
    elif exchange == 'BJ': return '2' + symbol

# 掘金格式转换 (如: 000001 → SZSE.000001)
def symbol_to_gmsymbol(symbol):
    if symbol[:2] in ['00', '30']: return f'SZSE.{symbol}'
    elif symbol[:2] in ['60', '68']: return f'SHSE.{symbol}'
```

### 3.2 股票分类识别系统

```python
# 文件: tools/utils_basic.py

def is_stock_10cm(code):        # 10%涨跌停限制
    return code[:2] in ['00', '60']

def is_stock_20cm(code):        # 20%涨跌停限制
    return code[:2] in ['30', '68']

def is_stock_cy(code):          # 创业板
    return code[:2] == '30'

def is_stock_kc(code):          # 科创板
    return code[:2] == '68'

def is_stock_bj(code):          # 北交所
    return code[:2] in ['82', '83', '87', '88', '43', '92']

def is_fund_etf(code):          # ETF基金
    return code[:2] in ['15', '51', '52', '53', '56', '58']
```

### 3.3 涨跌停价格计算

```python
def get_limit_up_price(code, pre_close):
    limit_rate = get_limiting_up_rate(code)
    limit = pre_close * limit_rate
    return float('%.2f' % limit)

def get_limiting_up_rate(code):
    if code[:2] in ['30', '68']: return 1.2    # 创业板/科创板 20%
    elif code[:1] in ['8', '9', '4']: return 1.3  # 北交所 30%
    else: return 1.1                           # 主板 10%
```

## 4. 技术指标计算引擎

### 4.1 MyTT技术指标库

项目集成了完整的MyTT技术指标计算库，支持主流技术分析指标：

```python
# 文件: mytt/MyTT.py

# 0级核心函数 - 基础计算
def MA(S, N):    # 简单移动平均
def EMA(S, N):   # 指数移动平均
def SMA(S, N, M=1):  # 中国式SMA
def STD(S, N):   # 标准差
def SUM(S, N):   # 累计和

# 1级应用函数 - 条件判断
def CROSS(S1, S2):      # 金叉判断
def COUNT(S, N):        # 条件计数
def BARSLAST(S):        # 上次条件成立周期数
def EXIST(S, N):        # 条件存在判断

# 2级技术指标 - 完整指标
def MACD(CLOSE, SHORT=12, LONG=26, M=9):
    DIF = EMA(CLOSE, SHORT) - EMA(CLOSE, LONG)
    DEA = EMA(DIF, M)
    MACD = (DIF - DEA) * 2
    return RD(DIF), RD(DEA), RD(MACD)

def KDJ(CLOSE, HIGH, LOW, N=9, M1=3, M2=3):
    LLN = LLV(LOW, N)
    RSV = (CLOSE - LLN) / (HHV(HIGH, N) - LLN) * 100
    K = EMA(RSV, (M1 * 2 - 1))
    D = EMA(K, (M2 * 2 - 1))
    J = K * 3 - D * 2
    return K, D, J
```

### 4.2 支持的技术指标清单

| 类别 | 指标名称 | 函数 | 用途 |
|------|---------|------|------|
| 趋势类 | MACD | `MACD()` | 趋势判断、背离分析 |
| | MA均线 | `MA()` | 趋势跟踪 |
| | EMA指数均线 | `EMA()` | 快速趋势 |
| 摆动类 | KDJ | `KDJ()` | 超买超卖 |
| | RSI | `RSI()` | 相对强弱 |
| | WR威廉指标 | `WR()` | 反转信号 |
| 波动类 | BOLL布林带 | `BOLL()` | 压力支撑 |
| | ATR真实波幅 | `ATR()` | 波动测量 |
| 成交量类 | OBV能量潮 | `OBV()` | 量价关系 |
| | MFI资金流 | `MFI()` | 资金强度 |

## 5. 数据缓存存储系统

### 5.1 历史数据缓存架构

#### 单例模式的DailyHistoryCache
```python
# 文件: reader/daily_history.py
class DailyHistoryCache:
    _instance = None
    daily_history = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DailyHistoryCache, cls).__new__(cls)
            cls.daily_history = None
        return cls._instance
```

#### DailyHistory核心功能
```python
class DailyHistory:
    default_init_day_count = 550  # 默认初始化550天数据
    default_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']
    
    def __init__(self, root_path='./_cache/_daily', init_day_count=550):
        self.root_path = root_path
        self.cache_history = {}  # 内存缓存
```

### 5.2 数据存储结构

#### 文件存储格式
```
_cache/
├── _daily/                    # 历史数据缓存
│   ├── _code_list.csv        # 股票代码列表
│   ├── 000001.SZ.csv         # 个股历史数据
│   ├── 000002.SZ.csv
│   └── ...
├── _code_names.csv           # 代码名称映射
└── _open_day_list_sina.csv   # 交易日历
```

#### 数据文件结构
```csv
datetime,open,high,low,close,volume,amount
20240101,10.50,10.80,10.30,10.75,1500000,161250000.00
20240102,10.75,11.00,10.60,10.90,1800000,194400000.00
```

### 5.3 缓存更新机制

#### 增量更新策略
```python
def download_single_daily(self, target_date):
    """单日增量更新"""
    code_list = self.get_code_list()
    updated_codes = self._download_date(target_date, code_list)
    
    # 批量更新缓存文件
    for code in updated_codes:
        self.cache_history[code] = self[code].sort_values(by='datetime')
        self.cache_history[code].to_csv(f'{self.root_path}/{code}.csv', index=False)

def download_recent_daily(self, days):
    """近期多日更新"""
    for forward_day in range(days, 0, -1):
        target_date = get_prev_trading_date(now, forward_day)
        sub_updated_codes = self._download_date(target_date, code_list)
```

#### 批量下载优化
```python
def _download_codes(self, code_list, day_count, data_source=DataSource.TUSHARE):
    group_size = 10  # 分组下载，避免API限制
    for i in range(0, len(code_list), group_size):
        group_codes = code_list[i:i + group_size]
        # 并发下载处理
```

### 5.4 多级缓存机制

#### 内存缓存层
```python
# tools/utils_cache.py
trade_day_cache = {}  # 交易日内存缓存

def check_is_open_day_sina(curr_date):
    # 1. 内存缓存检查
    if curr_date in trade_day_cache:
        return trade_day_cache[curr_date]
    
    # 2. 文件缓存检查
    if os.path.exists(TRADE_DAY_CACHE_PATH):
        trade_day_list = get_disk_trade_day_list_and_update_max_year()
        ans = curr_date in trade_day_list
        trade_day_cache[curr_date] = ans
        return ans
    
    # 3. 网络获取并缓存
    df = ak.tool_trade_date_hist_sina()
    df.to_csv(TRADE_DAY_CACHE_PATH)
```

#### 磁盘缓存管理
```python
# JSON缓存操作
def load_json(path):
    if os.path.exists(path):
        with open(path, 'r') as r:
            return json.loads(r.read())
    else:
        with open(path, 'w') as w:
            w.write('{}')
        return {}

def save_json(path, var):
    with open(path, 'w') as w:
        w.write(json.dumps(var, indent=4))

# Pickle缓存操作
def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
```

## 6. 数据质量保障

### 6.1 数据验证机制

```python
# 股票代码有效性验证
def is_stock(code_or_symbol):
    code_or_symbol = str(code_or_symbol)
    return code_or_symbol[:2] in ['00', '30', '60', '68', '82', '83', '87', '88', '43', '92']

# 数据完整性检查
def _download_codes(self, code_list, day_count, data_source):
    downloaded_count = 0
    download_failure = []
    
    for code in group_codes:
        df = get_daily_history(code, start_date, end_date, ...)
        if df is None or len(df) == 0:
            download_failure.append(code)
            continue
        else:
            df.to_csv(f'{self.root_path}/{code}.csv', index=False)
            downloaded_count += 1
```

### 6.2 异常处理机制

```python
# 网络异常重试
def get_ts_daily_history(code, start_date, end_date, ...):
    try_times = 0
    df = None
    while (df is None or len(df) <= 0) and try_times < 3:
        try_times += 1
        try:
            pro = get_tushare_pro()
            df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
        except Exception:
            continue
    return df

# 数据格式异常处理
try:
    df = pd.read_csv(path, dtype={'datetime': int})
    self.cache_history[code] = df
except Exception as e:
    print(code, e)
    error_count += 1
```

## 7. 实时数据处理

### 7.1 实时行情数据结构

```python
# tools/utils_remote.py
def quote_to_tick(quote):
    """实时行情转Tick数据"""
    ans = {
        'time': datetime.datetime.fromtimestamp(quote['time'] / 1000).strftime('%H:%M:%S'),
        'price': quote['lastPrice'],
        'volume': quote['volume'],
        'amount': quote['amount'],
    }
    
    # 五档买卖盘数据
    ap = adjust_list(quote['askPrice'], 5)
    ans.update({f"askPrice{i+1}": ap[i] for i in range(5)})
    
    av = adjust_list(quote['askVol'], 5)  
    ans.update({f"askVol{i+1}": av[i] for i in range(5)})
    
    bp = adjust_list(quote['bidPrice'], 5)
    ans.update({f"bidPrice{i+1}": bp[i] for i in range(5)})
    
    bv = adjust_list(quote['bidVol'], 5)
    ans.update({f"bidVol{i+1}": bv[i] for i in range(5)})
    
    return ans
```

### 7.2 数据更新频率控制

```python
# 当日K线数据更新
def quote_to_day_kline(quote, curr_date):
    return {
        'datetime': curr_date,
        'open': quote['open'],
        'high': quote['high'],
        'low': quote['low'],
        'close': quote['lastPrice'],
        'volume': quote['volume'],
        'amount': quote['amount'],
    }
```

## 8. 数据应用接口

### 8.1 选股器数据接口

```python
# selector模块调用示例
from reader.daily_history import DailyHistoryCache

def get_stock_data_for_selection():
    cache = DailyHistoryCache()
    daily_history = cache.daily_history
    
    # 获取指定股票池的历史数据
    subset_data = daily_history.get_subset_copy(codes=['000001.SZ', '000002.SZ'], days=20)
    
    return subset_data
```

### 8.2 交易模块数据接口

```python
# 获取实时行情用于交易决策
from tools.utils_mootdx import get_quotes

def get_realtime_data_for_trading(code_list):
    quotes = get_quotes(code_list)
    
    for code, quote in quotes.items():
        tick_data = quote_to_tick(quote)
        # 用于交易信号生成
```

## 9. 消息通知系统

### 9.1 钉钉通知
```python
# tools/utils_ding.py
class DingMessager:
    def send_text(self, text, output='', alert=False):
        res = self.send_message(data={
            "msgtype": "text",
            "text": {"content": text},
            "at": {"isAtAll": alert},
        })
        
    def send_markdown(self, title, text, output='', alert=False):
        # Markdown格式消息发送
```

### 9.2 飞书通知
```python
# tools/utils_feishu.py
class FeishuMessager:
    def send_markdown(self, title, text, output='', alert=False):
        card_style = get_feishu_markdown_card(title, text)
        my_data = {
            "timestamp": timestamp,
            "sign": sign,
            'msg_type': 'interactive',
            "card": card_style
        }
```

## 10. 性能优化策略

### 10.1 数据加载优化

1. **分批加载**: 将大量股票数据分组处理，避免内存溢出
2. **懒加载**: 只有在需要时才加载特定股票的历史数据
3. **内存管理**: 使用单例模式管理全局数据缓存

### 10.2 网络请求优化

1. **请求限流**: 分组下载避免API限制
2. **重试机制**: 网络异常时自动重试
3. **数据源切换**: 主备数据源自动切换

### 10.3 存储优化

1. **增量更新**: 只下载缺失的交易日数据
2. **压缩存储**: 合理的数据类型减少存储空间
3. **索引优化**: 按日期排序便于快速查询

## 11. 使用示例

### 11.1 获取历史数据
```python
from reader.daily_history import DailyHistoryCache

# 获取历史数据缓存实例
cache = DailyHistoryCache()
daily_history = cache.daily_history

# 获取单只股票数据
stock_data = daily_history['000001.SZ']
print(stock_data.tail())

# 获取多只股票最近20日数据
subset_data = daily_history.get_subset_copy(['000001.SZ', '000002.SZ'], 20)
```

### 11.2 更新数据
```python
# 更新单日数据
daily_history.download_single_daily('20241201')

# 更新最近5日数据
daily_history.download_recent_daily(5)

# 完整重新下载所有数据
daily_history.download_all_to_disk()
```

### 11.3 计算技术指标
```python
from mytt.MyTT import MACD, KDJ, RSI
import pandas as pd

# 获取收盘价数据
close_prices = stock_data['close'].values

# 计算MACD
dif, dea, macd = MACD(close_prices)

# 计算KDJ
k, d, j = KDJ(close_prices, stock_data['high'].values, stock_data['low'].values)

# 计算RSI
rsi = RSI(close_prices)
```

## 12. 总结

SilverQuant的数据处理与存储系统具有以下特点：

### 12.1 优势
1. **多数据源支持**: 支持AKShare、Tushare、MooTDX等多个数据源
2. **智能缓存机制**: 三级缓存提高数据访问效率
3. **完整技术指标库**: 集成MyTT库支持主流技术分析
4. **增量更新**: 高效的数据更新策略
5. **异常处理**: 完善的错误处理和重试机制
6. **格式转换**: 支持多种股票代码格式转换

### 12.2 应用场景
1. **历史回测**: 提供完整的历史数据支持
2. **实时交易**: 支持实时行情数据处理
3. **技术分析**: 丰富的技术指标计算
4. **选股研究**: 为选股算法提供数据基础
5. **风险控制**: 为风控系统提供数据支撑

这套数据处理系统为SilverQuant项目的各个交易策略和选股算法提供了坚实的数据基础，是整个量化交易系统的核心支撑模块。