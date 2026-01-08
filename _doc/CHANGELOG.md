# Changelog

所有关于本项目的显著变更都将记录在本文件中。

## [ To Do List ]

- LimitSeller 逻辑优化

## [ In Progress ]

### 添加
- 临跌停卖点模块，用以预防跌停被锁死的极端风险
- 竞价结束callback接口

### 修改
- 部分文案优化
- Subscriber代码拆分重构

### 删除
- 旧版schedule移除

## [ 4.2.0 ] 2025-11-18

### 添加
- Baostock 数据源添加
- Redis 数据策略分离的 Demo

### 修改
- AKShare 指数成份的缓存机制
- Daily Report 添加成本价显示
- 线程安全优化 @nackel
- TDX 除权细节优化 @dominicx
- apscheduler 稳定性增强 @dominicx

### 删除
- xtquant 静态库删除，改为 pip install

## [ 4.1.0 ] 2025-10-27

### 添加

- MiniQMT 的数据源支持
- TDX Zip 的数据源支持
- Tushare 数据源支持复权数据
- 增加临近盘前任务回调和检查
- backtest/gm_template 基于掘金的回测框架模版

### 修改

- Mootdx 部分小bug修复
- AKShare 添加本地缓存机制 AKCache 缓存部分接口
- AKShare 日线行情从东财改为sina减少被ban IP的概率
- Daily History 添加日内不重复更新机制
- Messager 美观度优化
- Subscriber 盘中Tick记录优化

### 删除

- 无

## [ 4.0.0 ] 2025-10-07

### 添加
- buyer: 添加批量买入逻辑
- buyer: 添加单日买入上限
- subscriber: 添加临近开盘的回调周期函数

### 修改
- held_info: 持仓记录支持不止天数的文件缓存结构
- daily_history: 优化了除权更新的逻辑

### 删除
- buyer: 无需再传入account_id
- runners: 移除旧版Scheduler定时器的样例实现
