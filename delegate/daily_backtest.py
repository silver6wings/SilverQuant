from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd


class Context:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __setattr__(self, name, value):
        # 禁止设置保留属性（双下划线开头和结尾）
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(f"Cannot set reserved attribute: {name}")
        super().__setattr__(name, value)

    def clear(self):
        self.__dict__.clear()


def backtest(
    daily_data: Dict[str, pd.DataFrame],
    stock_list: List[str],
    start_date: int,
    end_date: int,
    bars_count: int,
    handle_bars: Callable[[int, str, pd.DataFrame], None],
    before_day: Optional[Callable[[int], None]] = None,
    after_day: Optional[Callable[[int], None]] = None
) -> None:
    """
    高效回测框架 - 按时间顺序逐天遍历，处理局部数据缺失

    参数:
    data_dict: 股票数据字典 {股票代码: DataFrame}
    stock_list: 要处理的股票列表
    start_date: 回测开始日期 (整数格式YYYYMMDD)
    end_date: 回测结束日期 (整数格式YYYYMMDD)
    bar_count: 每次处理的K线数量
    hand_bars: 每只股票每根K线的回调函数
    before_day: 每日开始前的回调函数 (可选)
    after_day: 每日结束后的回调函数 (可选)
    """
    # 第一步：创建统一的交易日历
    all_dates = set()
    for df in daily_data.values():
        mask = (df['datetime'] >= start_date) & (df['datetime'] <= end_date)
        valid_dates = df.loc[mask, 'datetime'].unique()
        all_dates.update(valid_dates)

    # 转换为排序后的日期数组（升序）
    trade_dates = np.sort(np.array(list(all_dates)))

    # 第二步：为每只股票预构建索引映射
    stock_data = {}
    for stock in stock_list:
        if stock not in daily_data:
            continue

        df = daily_data[stock]
        # 筛选日期范围内的数据

        date_obj = datetime.strptime(str(start_date), '%Y%m%d')
        # 7天 / 5工作日 + 10天 长假
        new_date = date_obj - timedelta(days=int(bars_count * 1.5) + 10)
        begin_date = int(new_date.strftime('%Y%m%d'))
        mask = (df['datetime'] >= begin_date) & (df['datetime'] <= end_date)
        df_sub = df.loc[mask].copy()

        if df_sub.empty:
            continue

        # 确保按日期升序排列
        df_sub.sort_values('datetime', inplace=True)
        df_sub.reset_index(drop=True, inplace=True)

        # 创建日期到位置的映射
        date_to_idx = {date: idx for idx, date in enumerate(df_sub['datetime'])}

        # 存储处理后的数据
        stock_data[stock] = {
            'df': df_sub,
            'date_to_idx': date_to_idx,
            'min_idx': 0,
            'max_idx': len(df_sub) - 1
        }

    # 第三步：按时间顺序遍历每个交易日
    for current_date in trade_dates:
        # 盘前处理
        if before_day:
            before_day(current_date)

        # 处理当天的所有股票
        for stock in stock_list:
            if stock not in stock_data:
                continue

            data = stock_data[stock]
            df = data['df']
            date_to_idx = data['date_to_idx']

            # 检查当前日期是否存在
            if current_date not in date_to_idx:
                continue

            current_idx = date_to_idx[current_date]

            # 关键修改：从当前日期倒推bar_count条数据（处理停牌导致的局部缺失）
            # 计算窗口开始位置
            window_start_idx = current_idx - bars_count + 1

            # 如果窗口起始位置小于最小索引，则从最小索引开始
            if window_start_idx < data['min_idx']:
                window_start_idx = data['min_idx']

            # 获取窗口数据
            window = df.iloc[window_start_idx:current_idx + 1]

            # 调用K线处理函数
            if len(window) >= bars_count:
                handle_bars(current_date, stock, window.copy())

        # 盘后处理
        if after_day:
            after_day(current_date)
