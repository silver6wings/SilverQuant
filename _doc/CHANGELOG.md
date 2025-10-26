# Changelog

所有关于本项目的显著变更都将记录在本文件中。

## [ To do ]

### Add
- gm_template: 基于掘金的回测框架模版

### Modify
- 修复

### Remove
- 删除

## [ 4.0.0 ] - 2025-10-07

### Add
- buyer: 添加批量买入逻辑
- buyer: 添加单日买入上限
- subscriber: 添加临近开盘的回调周期函数

### Modify
- held_info: 持仓记录支持不止天数的文件缓存结构
- daily_history: 优化了除权更新的逻辑

### Remove
- buyer: 无需再传入account_id
- runners: 移除旧版Scheduler定时器的样例实现
