import json
import datetime
import time

import schedule
import threading
import os
import pandas as pd

from typing import Dict, Callable

from xtquant import xtdata

from delegate.xt_delegate import XtDelegate
from tools.utils_basic import code_to_symbol
from tools.utils_cache import check_today_is_open_day, get_total_asset_increase, \
    load_pickle, save_pickle, load_json, save_json, StockNames
from tools.utils_ding import DingMessager
from tools.utils_remote import get_ak_daily_history


class XtSubscriber:
    def __init__(
        self,
        account_id: str,
        strategy_name: str,
        delegate: XtDelegate,
        path_deal: str,
        path_assets: str,
        execute_strategy: Callable,     # 策略回调函数
        execute_interval: int = 1,      # 策略执行间隔，单位（秒）
        ding_messager: DingMessager = None,
        open_tick_memory_cache: bool = False,
        open_today_deal_report: bool = False,
        open_today_hold_report: bool = False,
    ):
        self.account_id = '**' + str(account_id)[-4:]
        self.strategy_name = strategy_name
        self.delegate = delegate

        self.path_deal = path_deal
        self.path_assets = path_assets

        self.execute_strategy = execute_strategy
        self.execute_interval = execute_interval
        self.ding_messager = ding_messager

        self.lock_quotes_update = threading.Lock()  # 聚合实时打点缓存的锁

        self.cache_quotes: Dict[str, Dict] = {}     # 记录实时的价格信息
        self.cache_limits: Dict[str, str] = {       # 限制执行次数的缓存集合
            'prev_seconds': '',                     # 限制每秒一次跑策略扫描的缓存
            'prev_minutes': '',                     # 限制每分钟屏幕心跳换行的缓存
        }
        self.cache_history: Dict[str, pd.DataFrame] = {}     # 记录历史日线行情的信息 { code: DataFrame }

        self.open_tick = open_tick_memory_cache
        self.quick_ticks: bool = False              # 是否开启quick tick模式
        self.today_ticks: Dict[str, list] = {}      # 记录tick的历史信息
        # [ 成交时间, 成交价格, 累计成交量 ]

        self.open_today_deal_report = open_today_deal_report
        self.open_today_hold_report = open_today_hold_report

        self.code_list = ['SH', 'SZ']
        self.stock_names = StockNames()
        self.last_callback_time = datetime.datetime.now()

    # -----------------------
    # 策略触发主函数
    # -----------------------
    def callback_sub_whole(self, quotes: Dict) -> None:
        now = datetime.datetime.now()
        self.last_callback_time = now

        curr_date = now.strftime('%Y-%m-%d')
        curr_time = now.strftime('%H:%M')

        # 每分钟输出一行开头
        if self.cache_limits['prev_minutes'] != curr_time:
            self.cache_limits['prev_minutes'] = curr_time
            print(f'\n[{curr_time}]', end='')

        curr_seconds = now.strftime('%S')
        with self.lock_quotes_update:
            self.cache_quotes.update(quotes)  # 合并最新数据

        if self.open_tick and (not self.quick_ticks):
            self.record_tick_to_memory(quotes)  # 更全（默认：先记录再执行）

        # 执行策略
        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            if int(curr_seconds) % self.execute_interval == 0:
                print('.' if len(self.cache_quotes) > 0 else 'x', end='')  # 每秒钟开始的时候输出一个点

                if self.execute_strategy(
                    curr_date,
                    curr_time,
                    curr_seconds,
                    self.cache_quotes,
                ):
                    with self.lock_quotes_update:
                        if self.open_tick and self.quick_ticks:
                            self.record_tick_to_memory(self.cache_quotes)  # 更快（先执行再记录）
                        self.cache_quotes.clear()  # execute_strategy() return True means need clear

    # -----------------------
    # 监测主策略执行
    # -----------------------
    def callback_monitor(self):
        now = datetime.datetime.now()

        if not check_today_is_open_day(now.strftime('%Y-%m-%d')):
            return

        if now - self.last_callback_time > datetime.timedelta(minutes=1):
            if self.ding_messager is not None:
                self.ding_messager.send_text(
                    f'[{self.account_id}]{self.strategy_name}:中断\n请检查QMT数据源 ',
                    alert=True,
                )
            if len(self.code_list) > 1 and xtdata.get_client():
                print('尝试重新订阅行情数据')
                time.sleep(1)
                self.unsubscribe_tick(notice=False)
                self.subscribe_tick(notice=False)

    # -----------------------
    # 订阅tick相关
    # -----------------------
    def subscribe_tick(self, notice=True):
        if not check_today_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.ding_messager is not None:
            self.ding_messager.send_text(f'[{self.account_id}]{self.strategy_name}:{"启动" if notice else "恢复"}')
        self.cache_limits['sub_seq'] = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        xtdata.enable_hello = False
        print('[启动行情订阅]', end='')

    def unsubscribe_tick(self, notice=True):
        if not check_today_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if 'sub_seq' in self.cache_limits:
            if self.ding_messager is not None:
                self.ding_messager.send_text(f'[{self.account_id}]{self.strategy_name}:{"关闭" if notice else "暂停"}')
            xtdata.unsubscribe_quote(self.cache_limits['sub_seq'])
            print('\n[关闭行情订阅]')

    def update_code_list(self, code_list: list[str]):
        # 防止没数据不打点
        code_list += ['000001.SH']
        self.code_list = code_list

    # -----------------------
    # 盘中实时的tick历史
    # -----------------------
    def record_tick_to_memory(self, quotes):
        # 记录 tick 历史
        for code in quotes:
            if code not in self.today_ticks:
                self.today_ticks[code] = []

            quote = quotes[code]
            tick_time = datetime.datetime.fromtimestamp(quote['time'] / 1000).strftime('%H:%M:%S')
            self.today_ticks[code].append([
                tick_time,                          # 成交时间，格式：%H:%M:%S
                round(quote['lastPrice'], 2),       # 成交价格
                round(quote['volume'], 0),          # 累计成交量（手）
                round(quote['askPrice'][0], 2),     # 卖一价格
                round(quote['askVol'][0], 2),       # 卖一数量
                round(quote['bidPrice'][0], 2),     # 买一价格
                round(quote['bidVol'][0], 2),       # 买一数量
            ])

    def clean_ticks_history(self):
        if not check_today_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        self.today_ticks.clear()
        self.today_ticks = {}
        print(f"已清除tick缓存")

    def save_tick_history(self):
        if not check_today_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return
        json_file = './_cache/debug/tick_history.json'
        with open(json_file, 'w') as file:
            json.dump(self.today_ticks, file, indent=4)
        print(f"当日tick数据已存储为 {json_file} 文件")

    # -----------------------
    # 盘前下载数据缓存
    # -----------------------
    def download_from_akshare(self, target_codes: list, start: str, end: str, adjust: str, columns: list[str]):
        print(f'Prepared time range: {start} - {end}')
        t0 = datetime.datetime.now()

        group_size = 200
        for i in range(0, len(target_codes), group_size):
            sub_codes = [sub_code for sub_code in target_codes[i:i + group_size]]
            time.sleep(1)
            print(i, sub_codes)  # 已更新数量
            for code in sub_codes:
                df = get_ak_daily_history(code, start, end, columns=columns, adjust=adjust)
                if df is not None:
                    self.cache_history[code] = df

        t1 = datetime.datetime.now()
        print(f'Prepared TIME COST: {t1 - t0}')

    def download_cache_history(
        self,
        cache_path: str,
        code_list: list[str],
        start: str,
        end: str,
        adjust: str,
        columns: list[str],
    ):
        temp_indicators = load_pickle(cache_path)
        if temp_indicators is not None and len(temp_indicators) > 0:
            # 如果有缓存就读缓存
            self.cache_history.clear()
            self.cache_history = {}
            self.cache_history.update(temp_indicators)
            print(f'{len(self.cache_history)} histories loaded from {cache_path}')
            if self.ding_messager is not None:
                self.ding_messager.send_text(f'[{self.account_id}]{self.strategy_name}:加载{len(self.cache_history)}支')
        else:
            # 如果没缓存就刷新白名单
            self.cache_history.clear()
            self.cache_history = {}
            self.download_from_akshare(code_list, start, end, adjust, columns)
            save_pickle(cache_path, self.cache_history)
            print(f'{len(self.cache_history)} of {len(code_list)} histories saved to {cache_path}')
            if self.ding_messager is not None:
                self.ding_messager.send_text(f'[{self.account_id}]{self.strategy_name}:下载{len(self.cache_history)}支')


    # -----------------------
    # 盘后报告总结
    # -----------------------
    def check_asset(self):
        curr_date = datetime.datetime.now().strftime('%Y-%m-%d')
        if not check_today_is_open_day(curr_date):
            return

        asset = self.delegate.check_asset()
        title = f'[{self.account_id}]{self.strategy_name} 盘后清点'
        txt = title

        increase = get_total_asset_increase(self.path_assets, curr_date, asset.total_asset)
        if increase is not None:
            txt += '\n>\n> '
            txt += f'当日变动: {"+" if increase > 0 else ""}{round(increase, 2)}元' \
                   f'({"+" if increase > 0 else ""}{round(increase * 100 / asset.total_asset, 2)}%)'

        txt += '\n>\n> '
        txt += f'持仓市值: {round(asset.market_value, 2)}元'

        txt += '\n>\n> '
        txt += f'剩余现金: {round(asset.cash, 2)}元'

        txt += f'\n>\n>'
        txt += f'资产总计: {round(asset.total_asset, 2)}元'

        if self.ding_messager is not None:
            self.ding_messager.send_markdown(title, txt)

    def today_deal_report(self):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if not check_today_is_open_day(today):
            return

        if not os.path.exists(self.path_deal):
            return

        if self.open_today_deal_report:
            df = pd.read_csv(self.path_deal, encoding='gbk')
            if '日期' in df.columns:
                df = df[df['日期'] == today]

            if len(df) > 0:
                title = f'{self.strategy_name} {today} 记录 {len(df)} 条'
                txt = title
                for index, row in df.iterrows():
                    # ['日期', '时间', '代码', '名称', '类型', '注释', '成交价', '成交量']
                    txt += '\n\n> '
                    txt += f'{row["时间"]} {row["注释"]} {row["代码"]} '
                    txt += '\n> '
                    txt += f'{row["名称"]} {row["成交量"]}股 {row["成交价"]}元 '

                if self.ding_messager is not None:
                    self.ding_messager.send_markdown(title, txt)

    def today_hold_report(self):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if not check_today_is_open_day(today):
            return

        if self.open_today_hold_report:
            positions = self.delegate.check_positions()
            txt = ''
            i = 0
            for position in positions:
                if position.volume > 0:
                    i += 1
                    txt += '\n\n>'
                    txt += f'' \
                           f'{code_to_symbol(position.stock_code)} ' \
                           f'{self.stock_names.get_name(position.stock_code)} ' \
                           f'{position.volume}股 ' \
                           f'{position.open_price:.2f}元'

            title = f'{self.strategy_name} {today} 持仓 {i} 支'
            txt = title + txt

            if self.ding_messager is not None:
                self.ding_messager.send_markdown(title, txt)

    # -----------------------
    # 定时器
    # -----------------------
    def start_scheduler(self):
        schedule.every().day.at('08:00').do(prev_check_open_day)

        if self.open_tick:
            schedule.every().day.at('09:10').do(self.clean_ticks_history)
            schedule.every().day.at('15:10').do(self.save_tick_history)

        schedule.every().day.at('09:25').do(self.subscribe_tick)
        schedule.every().day.at('11:30').do(self.unsubscribe_tick, False)

        schedule.every().day.at('13:00').do(self.subscribe_tick, False)
        schedule.every().day.at('15:00').do(self.unsubscribe_tick)

        schedule.every().day.at('15:01').do(self.today_deal_report)
        schedule.every().day.at('15:02').do(self.today_hold_report)
        schedule.every().day.at('15:03').do(self.check_asset)

        monitor_time_list = [
            '09:35', '09:45', '09:55', '10:05', '10:15', '10:25',
            '10:35', '10:45', '10:55', '11:05', '11:15', '11:25',
            '13:05', '13:15', '13:25', '13:35', '13:45', '13:55',
            '14:05', '14:15', '14:25', '14:35', '14:45', '14:55',
        ]
        for monitor_time in monitor_time_list:
            schedule.every().day.at(monitor_time).do(self.callback_monitor)


# -----------------------
# 检查是否交易日
# -----------------------
def prev_check_open_day():
    now = datetime.datetime.now()
    curr_date = now.strftime('%Y-%m-%d')
    curr_time = now.strftime('%H:%M')
    print(f'[{curr_time}]', end='')
    check_today_is_open_day(curr_date)


# -----------------------
# 持仓自动发现
# -----------------------
def update_position_held(lock: threading.Lock, delegate: XtDelegate, path: str):
    with lock:
        positions = delegate.check_positions()

        held_days = load_json(path)

        # 添加未被缓存记录的持仓
        for position in positions:
            if position.can_use_volume > 0:
                if position.stock_code not in held_days.keys():
                    held_days[position.stock_code] = 0

        if positions is not None and len(positions) > 0:
            # 删除已清仓的held_days记录
            position_codes = [position.stock_code for position in positions]
            print('当前持仓：', position_codes)
            holding_codes = list(held_days.keys())
            for code in holding_codes:
                if len(code) > 0 and code[0] != '_':
                    if code not in position_codes:
                        del held_days[code]

        save_json(path, held_days)


# -----------------------
# 订阅单个股票历史N个分钟级K线
# -----------------------
def sub_quote(
    callback: Callable,
    code: str,
    count: int = -1,
    period: str = '1m',
):
    xtdata.subscribe_quote(code, period=period, count=count, callback=callback)
