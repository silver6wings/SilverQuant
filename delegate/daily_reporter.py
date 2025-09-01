import os
from typing import Optional

import pandas as pd
from xtquant import xtdata

from delegate.xt_delegate import XtDelegate

from tools.utils_basic import code_to_symbol
from tools.utils_cache import StockNames
from tools.utils_ding import BaseMessager


def colour_text(text: str, to_red: bool, to_green: bool):
    color = '#3366FF'
    # （红色RGB为：220、40、50，绿色RGB为：22、188、80）
    if to_red:
        color = '#DC2832'
    if to_green:
        color = '#16BC50'

    return f'<font color="{color}">{text}</font>'


# 获取总仓位价格增幅
def get_total_asset_increase(path_assets: str, curr_date: str, curr_asset: float) -> Optional[float]:
    if os.path.exists(path_assets):
        df = pd.read_csv(path_assets)               # 读取
        prev_asset = df.tail(1)['asset'].values[0]  # 获取最近的日期资产
        df.loc[len(df)] = [curr_date, curr_asset]   # 添加最新的日期资产
        df.to_csv(path_assets, index=False)         # 存储
        return curr_asset - prev_asset
    else:
        df = pd.DataFrame({'date': [curr_date], 'asset': [curr_asset]})
        df.to_csv(path_assets, index=False)
        return None


class DailyReporter:
    def __init__(
        self,
        account_id: str,
        strategy_name: str,
        delegate: Optional[XtDelegate],
        path_deal: str,
        path_assets: str,
        messager: BaseMessager = None,
        use_outside_data: bool = False,     # 默认使用原版 QMT data （定期 call 数据但不传入quotes）
        today_report_show_bank: bool = False,   # 是否显示银行流水（国金QMT会卡死所以默认关闭）
    ):
        self.account_id = account_id
        self.strategy_name = strategy_name
        self.delegate = delegate

        self.path_deal = path_deal
        self.path_assets = path_assets

        self.messager = messager
        self.today_report_show_bank = today_report_show_bank
        self.use_outside_data = use_outside_data
        self.stock_names = StockNames()

    def today_deal_report(self, today: str):
        if not os.path.exists(self.path_deal):
            print('Missing deal record file!')
            title = f'[{self.account_id}]{self.strategy_name} 未找到记录'
            text = f'{title}\n\n[{today}] 未交易'
        else:
            df = pd.read_csv(self.path_deal, encoding='gbk')
            if '日期' in df.columns:
                df = df[df['日期'] == today]

            title = f'[{self.account_id}]{self.strategy_name} 委托统计'
            text = f'{title}\n\n[{today}] 交易{len(df)}单'

            if len(df) > 0:
                for index, row in df.iterrows():
                    # ['日期', '时间', '代码', '名称', '类型', '注释', '成交价', '成交量']
                    text += '\n\n> '
                    text += f'{row["时间"]} {row["注释"]} {code_to_symbol(row["代码"])} '
                    text += '\n>\n> '
                    text += f'{row["名称"]} {row["成交量"]}股 {row["成交价"]}元 '

        if self.messager is not None:
            self.messager.send_markdown(title, text)

    def today_hold_report(self, today: str, positions):
        text = ''
        hold_count = 0
        display_list = []

        # 处理持仓数据
        for position in positions:
            if position.volume > 0:
                code = position.stock_code
                if self.use_outside_data:
                    from tools.utils_remote import get_mootdx_quotes
                    quotes = get_mootdx_quotes([code])
                else:
                    quotes = xtdata.get_full_tick([code])

                curr_price = None
                if (code in quotes) and ('lastPrice' in quotes[code]):
                    curr_price = quotes[code]['lastPrice']

                open_price = position.open_price
                if open_price == 0.0 or curr_price is None:
                    continue

                vol = position.volume
                total_change = curr_price - open_price
                ratio_change = curr_price / open_price - 1
                hold_count += 1
                display_list.append([code, curr_price, vol, ratio_change, total_change])

        sorted_list_desc = sorted(display_list, key=lambda x: x[3], reverse=True)

        # 渲染输出内容
        for i in range(hold_count):
            [code, curr_price, vol, ratio_change, total_change] = sorted_list_desc[i]
            total_change = colour_text(f"{total_change * vol:.2f}", total_change > 0, total_change < 0)
            ratio_change = colour_text(f'{ratio_change * 100:.2f}%', ratio_change > 0, ratio_change < 0)

            text += '\n\n>'
            text += f'' \
                    f'{code_to_symbol(code)} ' \
                    f'{self.stock_names.get_name(code)} ' \
                    f'{curr_price * vol:.2f}元'
            text += '\n>\n>'
            text += f'盈亏比: {ratio_change} 盈亏额: {total_change}'

        title = f'[{self.account_id}]{self.strategy_name} 持仓统计'
        text = f'{title}\n\n[{today}] 持仓{hold_count}支\n{text}'

        if self.messager is not None:
            self.messager.send_markdown(title, text)

    def check_asset(self, today: str, asset):
        title = f'[{self.account_id}]{self.strategy_name} 盘后清点'
        text = title

        increase = get_total_asset_increase(self.path_assets, today, asset.total_asset)
        if increase is not None:
            text += '\n>\n> '

            total_change = colour_text(
                f'{"+" if increase > 0 else ""}{round(increase, 2)}',
                increase > 0,
                increase < 0,
                )
            ratio_change = colour_text(
                # (今日 - 昨日) / 昨日
                f'{"+" if increase > 0 else ""}{round(increase * 100 / (asset.total_asset - increase), 2)}%',
                increase > 0,
                increase < 0,
                )
            text += f'当日变动: {total_change}元({ratio_change})'

            if self.today_report_show_bank \
                    and hasattr(self.delegate, 'xt_trader') \
                    and hasattr(self.delegate.xt_trader, 'query_bank_info'):

                cash_change = 0.0
                today_xt = today.replace('-', '')
                bank_info = self.delegate.xt_trader.query_bank_info(self.delegate.account)  # 银行信息查询
                for bank in bank_info:
                    if bank.success:
                        # 银行卡流水记录查询
                        transfers = self.delegate.xt_trader.query_bank_transfer_stream(
                            self.delegate.account, today_xt, today_xt, bank.bank_no, bank.bank_account)
                        total_change = sum(
                            -t.balance
                            if t.transfer_direction == '2' else t.balance
                            for t in transfers if t.success
                        )
                        cash_change += total_change

                if abs(cash_change) > 0.0001:
                    cash_change = colour_text(
                        f'{"+" if cash_change > 0 else ""}{round(cash_change, 2)}',
                        cash_change > 0,
                        cash_change < 0,
                        )
                    text += '\n>\n> '
                    text += f'银证转账: {cash_change}元'

        text += '\n>\n> '
        text += f'持仓市值: {round(asset.market_value, 2)}元'

        text += '\n>\n> '
        text += f'剩余现金: {round(asset.cash, 2)}元'

        text += f'\n>\n>'
        text += f'资产总计: {round(asset.total_asset, 2)}元'

        if self.messager is not None:
            self.messager.send_markdown(title, text)
