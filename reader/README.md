# Multi-Source Reader

多数据源股票数据读取器，支持多个数据源的自动切换和统一接口。

## 概述

Multi-Source Reader 是一个统一的股票数据读取接口，支持多个数据源的自动切换和容错机制。当某个数据源不可用时，会自动尝试其他数据源，确保数据获取的稳定性。

## 支持的数据源

按默认优先级排序：

1. **XTData** - 迅投数据源
2. **AkShare** - 开源金融数据接口
3. **Tushare** - 财经数据接口
4. **Mootdx** - 通达信数据接口 (未实现)
5. **Amazing Data** - 奇点数据源 (未实现)

## 核心功能

### 主要接口

```python
from reader.multi_source_reader import read_daily_history

# 读取单只股票的日线数据
df = read_daily_history('000001.SZ', start_date='20240101', end_date='20241201')

# 读取多只股票的日线数据
data = read_daily_history(['000001.SZ', '000002.SZ'], start_date='20240101')

# 指定数据源优先级
df = read_daily_history('000001.SZ', sources_priority=['xtdata', 'akshare'])

# 指定复权方式
df = read_daily_history('000001.SZ', adjust='qfq')  # 前复权
```

### 参数说明

- `codes`: 股票代码，支持单个代码（字符串）或多个代码（列表）
- `start_date`: 开始日期，支持 'YYYYMMDD'、'YYYY-MM-DD' 格式或 datetime 对象
- `end_date`: 结束日期，格式同 start_date
- `sources_priority`: 数据源优先级列表，可选值：['xtdata', 'akshare', 'tushare', 'mootdx', 'amazing_data']
- `adjust`: 复权方式，'qfq'（前复权）、'hfq'（后复权）或 ''（不复权）

### 返回数据格式

- 单只股票：返回 pandas.DataFrame，包含列：datetime, open, high, low, close, volume, amount
- 多只股票：返回字典 {股票代码: DataFrame}

## 数据源配置

### 1. XTData 配置

**重要提示：在使用 XTData 前，需要先下载数据！**

XTData 需要本地安装迅投客户端并下载历史数据：

1. 安装迅投QMT客户端
2. 登录客户端并下载所需的历史数据
3. 确保 xtquant 模块正确安装和配置

```python
# XTData 使用示例
from xtquant import xtdata

# 确保数据已下载到本地
# 在客户端中下载历史数据后才能正常使用
```

### 2. Tushare 配置

**重要提示：需要在 tushare_token.py 中添加 ID 和 Token！**

1. 复制 `tushare_token_sample.py` 为 `tushare_token.py`
2. 在 [Tushare官网](https://tushare.pro/) 注册账号获取 token
3. 编辑 `tushare_token.py` 文件：

```python
# tushare_token.py
# 可以设置多个token
ts_token = [
    ['your_actual_token_here', 'your_user_id_here'],
    # 可以添加多个账号的token
    # ['another_token', 'another_id'],
]
```

**获取 Tushare Token 步骤：**
1. 访问 https://tushare.pro/register
2. 注册账号并完成邮箱验证
3. 登录后在个人中心获取 token
4. 将 token 和用户ID填入 `tushare_token.py`

### 3. 其他数据源

- **AkShare**: 无需配置，开箱即用 
- **Mootdx**: 无需配置，通过通达信接口获取数据 (未实现)
- **Amazing Data**: 根据具体配置要求设置 (未实现)

## 架构设计

### 适配器模式

每个数据源都实现了 `BaseAdapter` 接口：

```python
class BaseAdapter:
    def fetch_daily_history(self, code, start, end, options) -> Optional[pd.DataFrame]:
        """获取单只股票的日线数据"""
        
    def fetch_daily_history_batch(self, codes, start, end, options) -> Optional[Dict[str, pd.DataFrame]]:
        """批量获取多只股票的日线数据"""
        
    def supports_batch(self) -> bool:
        """是否支持批量获取"""
```

### 容错机制

1. **自动切换**: 当某个数据源失败时，自动尝试下一个数据源
2. **批量优化**: 优先使用支持批量获取的数据源
3. **错误日志**: 详细记录每个数据源的成功/失败状态

## 使用示例

### 基础用法

```python
from reader.multi_source_reader import read_daily_history

# 获取平安银行最近一年的数据
df = read_daily_history('000001.SZ', start_date='20231201', end_date='20241201')
print(df.head())

# 获取多只股票数据
stocks = ['000001.SZ', '000002.SZ', '600000.SH']
data = read_daily_history(stocks, start_date='20241101')
for code, df in data.items():
    print(f"{code}: {len(df)} records")
```

### 高级用法

```python
# 指定优先使用本地数据源
df = read_daily_history(
    '000001.SZ', 
    start_date='20240101',
    sources_priority=['xtdata', 'mootdx'],  # 优先本地数据源
    adjust='qfq'  # 前复权
)

# 批量获取并指定数据源
data = read_daily_history(
    ['000001.SZ', '000002.SZ', '600000.SH'],
    start_date='20241001',
    sources_priority=['akshare', 'tushare'],  # 优先网络数据源
)
```

## 注意事项

1. **数据源依赖**: 确保所需的数据源库已正确安装
2. **网络连接**: 网络数据源需要稳定的网络连接
3. **API限制**: 注意各数据源的API调用频率限制
4. **数据质量**: 不同数据源的数据可能存在细微差异
5. **本地数据**: XTData 需要预先下载数据到本地

## 故障排除

### 常见问题

1. **XTData 无数据**: 检查是否已在客户端下载历史数据
2. **Tushare 认证失败**: 检查 `tushare_token.py` 中的 token 配置
3. **网络超时**: 尝试切换到其他数据源或检查网络连接
4. **数据格式错误**: 检查股票代码格式是否正确（如 '000001.SZ'）

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用调试日志查看详细的数据源切换过程
df = read_daily_history('000001.SZ')
```

## 扩展开发

### 添加新数据源

1. 继承 `BaseAdapter` 类
2. 实现 `fetch_daily_history` 方法
3. 可选实现 `fetch_daily_history_batch` 方法
4. 在 `DEFAULT_ADAPTERS` 中注册新适配器

```python
class CustomAdapter(BaseAdapter):
    source = "custom"
    
    def fetch_daily_history(self, code, start, end, options):
        # 实现数据获取逻辑
        pass
```

## 许可证

请参考项目根目录的 LICENSE 文件。
