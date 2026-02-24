import os
import time
import datetime
import random
import threading
from typing import Dict, Callable, Optional

import pandas as pd
from xtquant import xtdata

from delegate.base_subscriber import HistorySubscriber
from delegate.xt_delegate import XtDelegate
from delegate.daily_reporter import DailyReporter

from tools.utils_cache import check_is_open_day
from tools.utils_ding import BaseMessager
from tools.utils_remote import qmt_quote_to_tick, qmt_pad_list


class XtSubscriber(HistorySubscriber):
    def __init__(
        self,
        # 基本信息
        account_id: str,
        delegate: Optional[XtDelegate],
        strategy_name: str,
        path_deal: str,
        path_assets: str,
        # 回调
        execute_strategy: Callable,         # 策略回调函数
        execute_call_end: Callable = None,  # 策略竞价结束回调
        execute_interval: int = 1,          # 策略执行间隔，单位（秒）
        before_trade_day: Callable = None,  # 盘前函数
        near_trade_begin: Callable = None,  # 盘后函数
        finish_trade_day: Callable = None,  # 盘后函数
        # 订阅
        use_ap_scheduler: bool = True,     # （已弃用）默认使用旧版 schedule
        # 通知
        ding_messager: BaseMessager = None,
        # 日报
        open_middle_end_report: bool = False,   # 午盘结束的报告
        open_today_deal_report: bool = False,   # 每日交易记录报告
        open_today_hold_report: bool = False,   # 每日持仓记录报告
        today_report_show_bank: bool = False,   # 是否显示银行流水（国金QMT会卡死所以默认关闭）
        # tick 缓存
        open_tick_memory_cache: bool = False,
        tick_memory_data_frame: bool = False,
    ):
        super().__init__(
            account_id=account_id,
            delegate=delegate,
            strategy_name=strategy_name,
            path_deal=path_deal,
            path_assets=path_assets,
            execute_strategy=execute_strategy,
            execute_call_end=execute_call_end,
            execute_interval=execute_interval,
            before_trade_day=before_trade_day,
            near_trade_begin=near_trade_begin,
            finish_trade_day=finish_trade_day,
            open_middle_end_report=open_middle_end_report,
            open_today_deal_report=open_today_deal_report,
            open_today_hold_report=open_today_hold_report,
            today_report_show_bank=today_report_show_bank,
            ding_messager=ding_messager,
        )
        self.use_ap_scheduler = use_ap_scheduler
        self.create_scheduler()

        self.open_tick = open_tick_memory_cache
        self.is_ticks_df = tick_memory_data_frame
        self.quick_ticks: bool = False                          # 是否开启quick tick模式
        self.today_ticks: Dict[str, list | pd.DataFrame] = {}   # 记录tick的历史信息
        self.lock_quotes_update = threading.Lock()  # 聚合实时打点缓存的锁

        self.cache_quotes: Dict[str, Dict] = {}     # 记录实时的价格信息
        self.sub_sequence: int = -1                 # 记录实时数据订阅号

        self.last_callback_time = datetime.datetime.now()       # 上次返回quotes 时间

        self.__extend_codes = ['399001.SZ', '399006.SZ', '159101.SZ', '159315.SZ', '159915.SZ',
                               '510500.SH', '510230.SH', '512680.SH', '588000.SH']

        self.code_list = ['000001.SH'] + self.__extend_codes

        self.daily_reporter = DailyReporter(
            account_id=self.account_id,
            delegate=self.delegate,
            strategy_name=self.strategy_name,
            path_deal=self.path_deal,
            path_assets=self.path_assets,
            messager=self.messager,
            use_outside_data=False,
            today_report_show_bank=self.today_report_show_bank,
        )

        # parquet/df 输出的统一列结构（无论 tick_memory_data_frame 开关如何，都需要用于落盘）
        self.tick_df_cols = ['local', 'time', 'price', 'high', 'low', 'lastClose', 'volume', 'amount'] \
            + [f'askPrice{i}' for i in range(1, 6)] \
            + [f'askVol{i}' for i in range(1, 6)] \
            + [f'bidPrice{i}' for i in range(1, 6)] \
            + [f'bidVol{i}' for i in range(1, 6)]

        self.curr_trade_date = '1990-12-19' #记录当前股票交易日期

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
            print(f'\n[{curr_date} {curr_time}]', end='')

        curr_seconds = now.strftime('%S')
        with self.lock_quotes_update:
            self.cache_quotes.update(quotes)  # 合并最新数据

        # 执行策略
        if self.cache_limits['prev_seconds'] != curr_seconds:
            self.cache_limits['prev_seconds'] = curr_seconds

            print_mark = "'" if int(curr_seconds) % 10 == 9 else "."
            print_mark = print_mark if len(self.cache_quotes) > 0 else "x"

            if int(curr_seconds) % self.execute_interval == 0:
                # 更全（默认：先记录再执行）
                if self.open_tick and (not self.quick_ticks):
                    self.record_tick_to_memory(self.cache_quotes)   # 这里用 cache_quotes 是防止积压导致丢数据

                # str(%Y-%m-%d) str(%H:%M) str(%S) dict(code: quotes)
                is_clear = self.execute_strategy(curr_date, curr_time, curr_seconds, self.cache_quotes)

                # 更快（先执行再记录）
                if self.open_tick and self.quick_ticks:
                    self.record_tick_to_memory(self.cache_quotes)   # 这里用 cache_quotes 是防止积压导致丢数据

                if is_clear:
                    with self.lock_quotes_update:
                        self.cache_quotes.clear()  # execute_strategy() return True means need clear

                print(print_mark, end='')  # 每秒钟开始的时候输出一个点

    # -----------------------
    # 监测主策略执行
    # -----------------------
    def callback_monitor(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        now = datetime.datetime.now()
        callback_timedelta = (now - self.last_callback_time).total_seconds()
        if callback_timedelta > 60:
            if self.messager is not None:
                self.messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name}:中断\n请检查QMT数据源 ',
                    alert=True,
                )
            if len(self.code_list) > 1 and xtdata.get_client():
                print('[恢复行情订阅] 断线尝试重连\n', end='')
                time.sleep(1)
                self.resubscribe_tick(notice=True)

    # -----------------------
    # 订阅 tick 相关
    # -----------------------
    def subscribe_tick(self, resume: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.messager is not None:
            self.messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name}:{"恢复" if resume else "开启"} '
                f'{len(self.code_list)}支',
                output='[Message] BEGIN SUBSCRIBING\n')
        xtdata.enable_hello = False
        self.sub_sequence = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)
        print(f'[开启行情订阅] 订阅数:{len(self.code_list)} 订阅号:{self.sub_sequence}', end='')

    def unsubscribe_tick(self, pause: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        if self.sub_sequence > 0:
            xtdata.unsubscribe_quote(self.sub_sequence)
            print(f'\n[结束行情订阅] 订阅数:{len(self.code_list)} 订阅号:{self.sub_sequence}\n', end='')
            if self.messager is not None:
                self.messager.send_text_as_md(
                    f'[{self.account_id}]{self.strategy_name}:{"暂停" if pause else "关闭"}',
                    output='[Message] END UNSUBSCRIBING\n')

    def resubscribe_tick(self, notice: bool = False):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        prev_sub_sequence = None
        if self.sub_sequence > 0:
            prev_sub_sequence = self.sub_sequence
            xtdata.unsubscribe_quote(self.sub_sequence)
        xtdata.enable_hello = False
        self.sub_sequence = xtdata.subscribe_whole_quote(self.code_list, callback=self.callback_sub_whole)

        if self.messager is not None and notice:
            self.messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name}:重启 {len(self.code_list)}支',
                output='[Message] FINISH RESUBSCRIBING\n')
        print(f'\n[重启行情订阅] 订阅数:{len(self.code_list)} 订阅号:{prev_sub_sequence} -> {self.sub_sequence}', end='')

    def update_code_list(self, code_list: list[str]):
        print(f'[订阅更新]:{code_list}\n', end='')
        # 加上证指数防止没数据不打点
        self.code_list = ['000001.SH'] + code_list
        extend = 10 - len(self.code_list)
        if extend > 0:
            self.code_list.extend(self.__extend_codes[:extend])  # 防止数据太少长时间不返回数据导致断流

    # -----------------------
    # 盘中实时的 tick 历史
    # -----------------------
    def record_tick_to_memory(self, quotes):
        # 记录 tick 历史到内存
        local_time = datetime.datetime.now().strftime('%H:%M:%S')
        if self.is_ticks_df:
            for code in quotes:
                tick = qmt_quote_to_tick(quotes[code])
                tick['local'] = local_time
                new_tick_df = pd.DataFrame([tick], columns=self.tick_df_cols)
                if code not in self.today_ticks:
                    self.today_ticks[code] = new_tick_df
                else:
                    self.today_ticks[code] = pd.concat([self.today_ticks[code], new_tick_df], ignore_index=True)
        else:
            for code in quotes:
                if code not in self.today_ticks:
                    self.today_ticks[code] = []

                quote = quotes[code]

                # 昨收：QMT字段历史上出现过 lastLose / lastClose 两种写法，这里做兼容，避免 KeyError
                last_close = quote.get('lastClose', None)
                if last_close is None:
                    last_close = quote.get('lastLose', 0) or 0
                last_close = float(last_close or 0.0)

                tick_ts_ms = quote.get('time', None)
                if isinstance(tick_ts_ms, (int, float)) and tick_ts_ms > 0:
                    tick_time = datetime.datetime.fromtimestamp(tick_ts_ms / 1000).strftime('%H:%M:%S')
                else:
                    tick_time = datetime.datetime.now().strftime('%H:%M:%S')

                ask_price = qmt_pad_list(quote.get('askPrice', []), target_length=5, fill=0.0)
                ask_vol = qmt_pad_list(quote.get('askVol', []), target_length=5, fill=0)
                bid_price = qmt_pad_list(quote.get('bidPrice', []), target_length=5, fill=0.0)
                bid_vol = qmt_pad_list(quote.get('bidVol', []), target_length=5, fill=0)
                self.today_ticks[code].append([
                    tick_time,                                          # 成交时间，格式：%H:%M:%S
                    round(float(quote.get('lastPrice', 0) or 0), 3),    # 成交价格
                    round(float(quote.get('high', 0) or 0), 3),         # 成交最高价
                    round(float(quote.get('low', 0) or 0), 3),          # 成交最低价
                    round(last_close, 3),                               # 昨日收盘价
                    int(quote.get('volume', 0) or 0),                   # 累计成交量（手）
                    round(float(quote.get('amount', 0) or 0), 3),       # 累计成交额（元）
                    [round(float(p or 0.0), 3) for p in ask_price],     # 卖价
                    [int(v or 0) for v in ask_vol],                     # 卖量
                    [round(float(p or 0.0), 3) for p in bid_price],     # 买价
                    [int(v or 0) for v in bid_vol],                     # 买量
                    local_time,                                         # 本机记录时间（local）
                ])

    def clean_ticks_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        self.today_ticks.clear()
        self.today_ticks = {}
        print(f"[提示] 已清除tick缓存")

    def save_tick_history(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        out_dir = './_cache/debug'
        os.makedirs(out_dir, exist_ok=True)
        parquet_file = f'{out_dir}/ticks_{self.strategy_name}_{datetime.datetime.now().strftime("%A").lower()}.parquet'

        frames: list[pd.DataFrame] = []
        if self.is_ticks_df:
            for code, df in (self.today_ticks or {}).items():
                if df is None or len(df) == 0:
                    continue
                code_df = df.copy()
                code_df.insert(0, 'code', code)
                frames.append(code_df)
        else:
            for code, ticks in (self.today_ticks or {}).items():
                if not isinstance(ticks, list) or len(ticks) == 0:
                    continue

                # list 模式也拍平为与 df 模式一致的列结构（包含盘口 1~5 档），方便统一读写
                records: list[dict] = []
                err = []
                for t in ticks:
                    # t: [time, price, high, low, lastClose, volume, amount, askPrice(list5), askVol(list5), bidPrice(list5), bidVol(list5), local]
                    try:
                        ask_p = list(t[7]) if isinstance(t[7], (list, tuple)) else []
                        ask_v = list(t[8]) if isinstance(t[8], (list, tuple)) else []
                        bid_p = list(t[9]) if isinstance(t[9], (list, tuple)) else []
                        bid_v = list(t[10]) if isinstance(t[10], (list, tuple)) else []
                    except Exception as e:
                        err.append([str(t[0]), str(e)])
                        ask_p, ask_v, bid_p, bid_v = [], [], [], []

                    # list 模式新结构：末尾追加 local（本机记录时间）；旧结构缺失时退化为与 time 相同
                    local = t[11] if (hasattr(t, "__len__") and len(t) > 11) else t[0]
                    rec = {
                        'local': local,
                        'time': t[0],
                        'price': t[1],
                        'high': t[2],
                        'low': t[3],
                        'lastClose': t[4],
                        'volume': t[5],
                        'amount': t[6],
                    }
                    for i in range(5):
                        rec[f'askPrice{i+1}'] = ask_p[i] if i < len(ask_p) else 0.0
                        rec[f'askVol{i+1}'] = ask_v[i] if i < len(ask_v) else 0
                        rec[f'bidPrice{i+1}'] = bid_p[i] if i < len(bid_p) else 0.0
                        rec[f'bidVol{i+1}'] = bid_v[i] if i < len(bid_v) else 0
                    records.append(rec)

                if len(err) > 0:
                    print(f'[提示] {code} 的 tick 本地持久化时报错：{err}')

                code_df = pd.DataFrame.from_records(records, columns=self.tick_df_cols)
                code_df.insert(0, 'code', code)
                frames.append(code_df)

        if not frames:
            print(f"[提示] 当日tick数据为空，未写入 {parquet_file}")
            return

        all_df = pd.concat(frames, ignore_index=True)
        # parquet 写入依赖 pyarrow/fastparquet；本项目已在 requirements.txt 补充 pyarrow
        # zstd 在极少数环境可能不可用，这里做一次降级，确保“必然能写出文件”。
        try:
            all_df.to_parquet(parquet_file, index=False, compression='zstd', engine='pyarrow')
        except Exception as e:
            print(f"[提示] 当日tick数据zstd压缩存储失败，改用snappy压缩存储：", e)
            all_df.to_parquet(parquet_file, index=False, compression='snappy', engine='pyarrow')
        print(f"[提示] 当日tick数据已存储为 {parquet_file} 文件 rows={len(all_df)}")

    # -----------------------
    # 定时器
    # -----------------------

    def clear_all(self):
        super().clear_all()
        self.cache_quotes.clear()
        self.cache_history.clear()
        self.today_ticks.clear()

    def before_trade_day_wrapper(self):
        if not check_is_open_day(datetime.datetime.now().strftime('%Y-%m-%d')):
            return

        self.clear_all()
        self.code_list = ['000001.SH'] + self.__extend_codes

        if self.before_trade_day is not None:
            self.before_trade_day()
            self.curr_trade_date = datetime.datetime.now().strftime('%Y-%m-%d')
            print(f'[定时任务] 盘前准备完成 {self.curr_trade_date}\n', end='')


    def _start_qmt_scheduler(self):
        # 默认定时任务列表
        cron_jobs = [
            ['01:00', self.prev_check_open_day, None],
            ['08:30', self.near_trade_begin_wrapper, None],
            ['08:55', self.check_before_finished, None],
            ['09:14', self.subscribe_tick, None],
            ['11:31', self.unsubscribe_tick, (True, )],
            ['12:59', self.subscribe_tick, (True, )],
            ['15:01', self.unsubscribe_tick, None],
            ['15:02', self.daily_summary, None],
        ]

        if self.open_tick:
            cron_jobs.append(['09:10', self.clean_ticks_history, None])
            cron_jobs.append(['15:10', self.save_tick_history, None])

        if self.before_trade_day is not None:
            # random 时间为了跑多个策略时防止短期预加载数据流量压力过大
            before_time = f'0{random.randint(0, 3) + 3}:{random.randint(0, 59)}'  # 03:00 ~ 06:59
            cron_jobs.append([before_time, self.before_trade_day_wrapper, None])

        if self.finish_trade_day is not None:
            # random 时间为了跑多个策略时防止短期预加载数据流量压力过大
            finish_time = f'16:{random.randint(0, 10) + 5}'  # 16:05 ~ 16:15
            cron_jobs.append([finish_time, self.finish_trade_day_wrapper, None])

        if self.execute_call_end is not None:
            cron_jobs.append(['09:26', self.execute_call_end_wrapper, None])

        if self.open_middle_end_report:
            cron_jobs.append(['11:32', self.daily_summary, None])

        # 新版 apscheduler
        for cron_job in cron_jobs:
            [hr, mn] = cron_job[0].split(':')
            if cron_job[2] is None:
                self.scheduler.add_job(cron_job[1], 'cron', hour=hr, minute=mn)
            else:
                self.scheduler.add_job(cron_job[1], 'cron', hour=hr, minute=mn, args=list(cron_job[2]))

        # 尝试重新订阅 tick 数据，减少30分时无数据返回机率
        self.scheduler.add_job(self.resubscribe_tick, 'cron', hour=9, minute=29, second=30)

        # 数据源中断检查时间点
        monitor_time_list = [
            '09:35', '09:45', '09:55', '10:05', '10:15', '10:25',
            '10:35', '10:45', '10:55', '11:05', '11:15', '11:25',
            '13:05', '13:15', '13:25', '13:35', '13:45', '13:55',
            '14:05', '14:15', '14:25', '14:35', '14:45', '14:55',
        ]

        for monitor_time in monitor_time_list:
            [hr, mn] = monitor_time.split(':')
            self.scheduler.add_job(self.callback_monitor, 'cron', hour=hr, minute=mn)

        # 启动定时器
        try:
            print('[定时任务] 计划启动')
            self.scheduler.start()
        except KeyboardInterrupt:
            print('[定时任务] 手动结束')
            os.system('pause')
        except Exception as e:
            print('[定时任务] 执行出错：', e)
            os.system('pause')
        finally:
            self.delegate.shutdown()
            print('[定时任务] 关闭完成')
            try:
                import sys
                sys.exit(0)
            except SystemExit:
                os._exit(0)


    def start_scheduler(self):
        temp_now = datetime.datetime.now()
        temp_date = temp_now.strftime('%Y-%m-%d')
        temp_time = temp_now.strftime('%H:%M')
        # 盘中执行需要补齐
        if '08:05' < temp_time < '15:30' and check_is_open_day(temp_date):
            self.prev_check_open_day()
            self.before_trade_day_wrapper()
            self.near_trade_begin_wrapper()
            if '09:15' < temp_time < '11:30' or '13:00' <= temp_time < '14:57':
                self.subscribe_tick()  # 重启时如果在交易时间则订阅Tick

        self._start_qmt_scheduler()


# -----------------------
# 临时获取quotes
# -----------------------
def xt_get_ticks(code_list: list[str]) -> dict[str, dict]:
    # http://docs.thinktrader.net/pages/36f5df/#%E8%8E%B7%E5%8F%96%E5%85%A8%E6%8E%A8%E6%95%B0%E6%8D%AE
    return xtdata.get_full_tick(code_list)
