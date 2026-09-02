# 大 QMT 桥接

> 适用场景：券商仍提供 **大 QMT（完整客户端）**，但 **MiniQMT / `userdata_mini` 不可用** 或无法作为本地 IPC 交易网关时，用本方案把 SilverQuant 接到大 QMT 上跑实盘/模拟。

> 更详细的 helper 部署说明可参考：[BulletTrade 大 QMT 文档](https://bullettrade.cn/docs/big-qmt-server.html#6-helper)

---

## 与 MiniQMT 的区别

| 维度 | MiniQMT（默认） | 大 QMT 桥接 |
| --- | --- | --- |
| 依赖 | 券商 `userdata_mini` + 原生 `xtquant` | 大 QMT 客户端 + `btquant` + HTTP helper |
| 配置开关 | `credentials.USE_BIG_QMT = False` | `credentials.USE_BIG_QMT = True` |
| 策略代码 | `from tools.utils_xtquant import ...` | 同上，无需改业务 import |
| 行情推送 | 真全推 / 低延迟 | HTTP 轮询 tick（默认 1 秒） |
| 历史日线缓存 | `DailyHistoryXT` 可用 | **不可用**，请用 Tushare / MOOTDX |

SilverQuant 的 `XtDelegate`、`XtSubscriber`、`run_wencai_qmt.py` 等入口 **无需修改**；由 `tools/utils_xtquant.py` 根据 `USE_BIG_QMT` 自动加载 `btquant` 或原生 `xtquant`。

---

## 架构

```
SilverQuant 策略（Python）
    │  tools/utils_xtquant → btquant
    │  HTTP 127.0.0.1:9000
    ▼
大 QMT 策略 BIGQMTHELPER（btquant/big_qmt_gateway.py）
    │  ContextInfo / passorder
    ▼
券商 QMT 账户（实盘 / 模拟）
```

---

## 能力说明

### 已支持（可跑主策略）

| 能力 | 说明 |
| --- | --- |
| 连接 / 订阅 | `connect` / `subscribe`，网关健康检查 |
| 账户查询 | 资金、可用、市值、总资产 |
| 持仓查询 | 代码、数量、可用、成本、市值 |
| 委托 / 成交 | 当日委托列表、成交列表、单笔委托查询 |
| 限价 / 市价下单 | 限价、深市/沪市市价；miniqmt 的 `price_type` 会自动映射为大 QMT `passorder` 的 `prType` |
| 撤单 | 按 `order_id` 撤单 |
| 行情快照 | `get_full_tick` |
| Tick 订阅 | `subscribe_whole_quote`（HTTP 轮询，需传**具体股票代码**） |
| 交易回调 | 委托/成交/下单响应（轮询模式，非毫秒级推送） |

### 部分支持 / 有差异

| 能力 | 说明 |
| --- | --- |
| 全推行情 | 不能像 MiniQMT 那样传 `['SH','SZ']` 订全市场；只能按代码列表轮询 |
| 推送延迟 | 由 `qmt_bridge.json` 的 `tick_poll_interval_seconds` 控制，默认 1 秒 |
| 市价单 | 深市 miniqmt `17` 会映射为 passorder `47`；保护价由最新价 + `market_order_premium` 或 0（涨跌停）计算 |
| 委托号 | 大 QMT 有时 `passorder` 先返回空 id，需稍后 `query_stock_orders` 再查 |

### 不支持

| 能力 | 说明 |
| --- | --- |
| `DailyHistoryXT` | 依赖 miniqmt 本地缓存与 `get_market_data_ex`；`USE_BIG_QMT=True` 时会提示，请用 Tushare / MOOTDX |
| `subscribe_quote` | K 线订阅未实现 |
| 新股申购 | `query_ipo_data` 等返回空 |
| 银证转账查询 | `query_bank_info` 未实现 |
| MiniQMT 路径 | `QMT_CLIENT_PATH` / `userdata_mini` 在桥接模式下不使用 |

---

## 需要的文件

复制或保留在项目根目录：

| 文件 / 目录 | 作用 |
| --- | --- |
| `btquant/` | 大 QMT 桥接影子包（API 与 xtquant 同名） |
| `qmt_bridge.json` | 网关地址、密码、账号、轮询间隔等 |
| `credentials.py` | 设置 `USE_BIG_QMT = True` 及 `QMT_ACCOUNT_ID` |

说明文档：`_doc/BIG_QMT.md`（本页）

---

## 部署步骤

### 1. 在大 QMT 中创建 helper 策略

1. **模型研究** → 新建 Python 策略，名称建议：`BIGQMTHELPER`
2. 将 `btquant/big_qmt_gateway.py` **全文粘贴**到大 QMT 策略编辑器
3. 修改文件顶部参数（**不要与 QMT 登录密码相同**）：
   - `GATEWAY_PASSWORD`
   - `GATEWAY_SECRET`
   - `ACCOUNT_ID`（资金账号）
4. 确认 `ENABLE_TRADING = True`（需要下单时）

### 2. 启动模型交易

1. **模型交易** → 新建运行项，选择 `BIGQMTHELPER`
2. 建议参数：
   - 账号类型：股票账户
   - 资金账号：`ACCOUNT_ID`
   - 主图代码：`000300`
   - 运行周期：日线
   - **勾选「终端启动后自动运行」**，延迟 **10 秒**
   - **不要勾选「启动本地 Python」**
3. 查看 **策略日志**，正常应出现类似：
   ```
   [BT_BIG_QMT] listen success listen=127.0.0.1:9000
   ```

### 3. 配置 SilverQuant

1. **`qmt_bridge.json`**：与 helper 保持一致

   ```json
   {
     "gateway_url": "http://127.0.0.1:9000",
     "gateway_password": "与 GATEWAY_PASSWORD 一致",
     "gateway_secret": "与 GATEWAY_SECRET 一致",
     "account_id": "资金账号",
     "account_type": "stock",
     "tick_poll_interval_seconds": 1,
     "market_order_premium": 0.02
   }
   ```

2. **`credentials.py`**：

   ```python
   USE_BIG_QMT = True
   QMT_ACCOUNT_ID = '与 qmt_bridge.json account_id 一致'
   ```

3. 正常启动策略，例如：`run_wencai_qmt.py`（`IS_PROD = True` 时为实盘）

### 4. 验证

在项目根目录运行：

```bash
python testcase/try_xtquant.py
```

可检查：连接、资金、持仓、委托、成交、快照；下单示例在文件内注释，**慎用实盘**。

---

## 常见问题

**日志只有 `module loaded` 就停了**

未进入 `init(ContextInfo)`。确认从 **模型交易** 启动，且未勾选「启动本地 Python」。

**`order_stock` 返回 0 但看不到委托**

大 QMT 可能延迟返回委托号；等待数秒后调用 `query_stock_orders`，或看 helper 策略日志里的 `passorder` 输出。

**市价单失败**

确认 helper 已 `ENABLE_TRADING = True`；交易时段内再试；深市 ETF 使用 `MARKET_SZ_CONVERT_5_CANCEL`（桥接层会映射为 prType 47）。

**PyCharm 报找不到 xtquant**

大 QMT 模式应走 `btquant`，设置 `USE_BIG_QMT = True`，并将项目根目录 Mark as Sources Root。

**历史日线下载失败**

预期行为。`USE_BIG_QMT=True` 时不要使用 `DailyHistoryXT`，改用 Tushare 或 MOOTDX 数据源。

---

## 相关代码

| 路径 | 说明 |
| --- | --- |
| `tools/utils_xtquant.py` | `USE_BIG_QMT` 开关与统一 import |
| `btquant/xttrader.py` | 交易代理、price_type 映射 |
| `btquant/xtdata.py` | 行情代理、tick 轮询 |
| `btquant/big_qmt_gateway.py` | 粘贴到大 QMT 的 helper 源码 |
| `testcase/try_xtquant.py` | 连接 / 查询 / 下单示例 |

---

## 参考与鸣谢

本项目的 `btquant` 大 QMT 桥接方案在协议与 helper 思路上参考了 **[BulletTrade](https://github.com/BulletTrade/bullet-trade)** 的大 QMT 网关设计，在此向 BulletTrade 作者及社区表示感谢。

**配置与 helper 详细说明**（大 QMT 策略粘贴、网关参数、listen success 等）请参阅官方文档：

> [BulletTrade 大 QMT 服务 — Helper 部署](https://bullettrade.cn/docs/big-qmt-server.html#6-helper)

---

## 对下单速度有更高要求？

`btquant` 走 **HTTP 轮询 + passorder**，适合 SilverQuant 常规实盘；若策略对 **委托延迟、推送实时性** 要求更极致，可评估 **[cfquant](https://github.com/95ge/cfquant)**：同样把大 QMT 转为 `xtquant` 兼容层，并提供 **低延迟交易桥**（`CFQUANT_TRADE_LOWLAT.py`）与本地 LTtx 通信，延迟通常低于 HTTP 网关方案。

> cfquant 仓库：<https://github.com/95ge/cfquant>

两者可并存选型：`btquant` 与 SilverQuant 已深度集成、开箱即用；cfquant 更适合自行对接、以速度为优先的场景。
