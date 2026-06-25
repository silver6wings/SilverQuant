"""
掘金交易回调入口。

gmtrade SDK 在 Windows 下会把 delegate/ 路径误判为 D: 盘符（No module named 'elegate'），
回调函数通过此文件暴露，由 gm_callback.register_callback() 加载。
"""
from delegate.gm_callback import (
    on_account_status,
    on_error,
    on_execution_report,
    on_order_status,
    on_trade_data_connected,
    on_trade_data_disconnected,
)

__all__ = [
    'on_account_status',
    'on_error',
    'on_execution_report',
    'on_order_status',
    'on_trade_data_connected',
    'on_trade_data_disconnected',
]
