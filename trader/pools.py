import pandas as pd
from typing import Set, Callable

from tools.utils_basic import symbol_to_code
from tools.utils_cache import get_prefixes_stock_codes, get_index_constituent_codes
from tools.utils_remote import get_wencai_codes, get_tdx_zxg_code

from trader.pools_indicator import get_macd_index_indicator, get_ma_index_indicator
from trader.pools_section import get_dfcf_industry_stock_codes, get_dfcf_industry_sections, \
    get_ths_concept_sections, get_ths_concept_stock_codes


class StockPool:
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        self.account_id = '**' + str(account_id)[-4:]
        self.strategy_name = strategy_name
        self.ding_messager = ding_messager

        self.pool_parameters = parameters
        self.cache_blacklist: Set[str] = set()
        self.cache_whitelist: Set[str] = set()

    def get_code_list(self) -> list[str]:
        return list(self.cache_whitelist.difference(self.cache_blacklist))

    def refresh(self):
        self.refresh_black()
        self.refresh_white()

        print(f'[POOL] White list refreshed {len(self.cache_whitelist)} codes.')
        print(f'[POOL] Black list refreshed {len(self.cache_blacklist)} codes.')
        print(f'[POOL] Total list refreshed {len(self.get_code_list())} codes.')

        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(
                f'{self.strategy_name}:股票池{len(self.get_code_list())}支\n'
                f'白名单: {len(self.cache_whitelist)} 黑名单: {len(self.cache_blacklist)}')

    def refresh_black(self):
        self.cache_blacklist.clear()

    def refresh_white(self):
        self.cache_whitelist.clear()

    # 删除不符合模式和没有缓存的票池
    def filter_white_list_by_selector(self, filter_func: Callable, cache_history: dict[str, pd.DataFrame]):
        remove_list = []
        print('[POOL] Filtering...', end='')

        i = 0
        for code in self.cache_whitelist:
            i += 1
            if i % 200 == 0:
                print(f'{i}.', end='')
            if code in cache_history:
                try:
                    df = filter_func(cache_history[code], code, None)  # 预筛公式默认不需要使用quote所以传None
                    if (len(df) > 0) and (not df['PASS'].values[-1]):
                        remove_list.append(code)
                except Exception as e:
                    print(f'[POOL] Drop {code} when filtering: ', e)
                    remove_list.append(code)
            else:
                remove_list.append(code)

        for code in remove_list:
            self.cache_whitelist.discard(code)

        print(f'[POOL] {len(remove_list)} codes filter out.')

        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(f'[{self.account_id}]{self.strategy_name}:筛除{len(remove_list)}支\n')


# -----------------------
# Black Empty
# -----------------------

class StocksPoolBlackEmpty(StockPool):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)


# -----------------------
# Black Wencai
# -----------------------

class StocksPoolBlackWencai(StockPool):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.black_prompts = parameters.black_prompts

    def refresh_black(self):
        super().refresh_black()

        codes = get_wencai_codes(self.black_prompts)
        self.cache_blacklist.update(codes)


# -----------------------
# White Wencai
# -----------------------

class StocksPoolWhiteWencai(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_prompts = parameters.white_prompts

    def refresh_white(self):
        super().refresh_white()

        codes = get_wencai_codes(self.white_prompts)
        self.cache_whitelist.update(codes)


# -----------------------
# White Custom
# -----------------------

# 自定义白名单股票列表
class StocksPoolWhiteCustomSymbol(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_codes_filepath = parameters.white_codes_filepath

    def refresh_white(self):
        super().refresh_white()

        with open(self.white_codes_filepath, 'r') as r:
            lines = r.readlines()
            codes = []
            for line in lines:
                line = line.replace('\n', '')
                if len(line) >= 6:
                    line = line[-6:]  # 只获取最后六位
                    code = symbol_to_code(line)
                    codes.append(code)
            self.cache_whitelist.update(codes)


class StocksPoolWhiteCustomTdx(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        # 自选股文件默认路径示例： r'C:\new_tdx\T0002\blocknew\ZXG.blk'
        self.tdx_codes_filepath = parameters.tdx_codes_filepath

    def refresh_white(self):
        super().refresh_white()
        codes = get_tdx_zxg_code(self.tdx_codes_filepath)
        self.cache_whitelist.update(codes)


# -----------------------
# White Indexes
# -----------------------

# 自定义指数成份股
class StocksPoolWhiteIndexes(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_indexes = parameters.white_indexes

    def refresh_white(self):
        super().refresh_white()

        for index in self.white_indexes:
            t_white_codes = get_index_constituent_codes(index)
            self.cache_whitelist.update(t_white_codes)


# 自定义指数成份股 + 指数MA择时
class StocksPoolWhiteIndexesMA(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_indexes = parameters.white_prefixes
        self.white_index_symbol = parameters.white_index_symbol         # 指数名称（默认中证全指000985）
        self.white_ma_above_period = parameters.white_ma_above_period   # 均线周期（默认五日均线）

    def refresh_white(self):
        super().refresh_white()

        allow, info = get_ma_index_indicator(
            symbol=self.white_index_symbol,
            period=self.white_ma_above_period,
        )
        if allow:
            for index in self.white_indexes:
                t_white_codes = get_index_constituent_codes(index)
                self.cache_whitelist.update(t_white_codes)


# 自定义指数成份股 + 指数群MACD择时
class StocksPoolWhiteIndexesMACD(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_indexes = parameters.white_indexes

    def refresh_white(self):
        super().refresh_white()

        for index in self.white_indexes:
            allow, info = get_macd_index_indicator(symbol=index)
            if allow:
                t_white_codes = get_index_constituent_codes(index)
                self.cache_whitelist.update(t_white_codes)


# -----------------------
# White Prefixes
# -----------------------

# 自定义前缀成份股
class StocksPoolWhitePrefixes(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_prefixes = parameters.white_prefixes
        if hasattr(parameters, 'white_none_st'):
            self.white_none_st = parameters.white_none_st
        else:
            self.white_none_st = False

    def refresh_white(self):
        super().refresh_white()

        t_white_codes = get_prefixes_stock_codes(self.white_prefixes, self.white_none_st)
        self.cache_whitelist.update(t_white_codes)


# 自定义前缀成份股 + 指数MA择时
class StocksPoolWhitePrefixesMA(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_prefixes = parameters.white_prefixes
        self.white_index_symbol = parameters.white_index_symbol         # 指数名称（默认中证全指000985）
        self.white_ma_above_period = parameters.white_ma_above_period   # 均线周期（默认五日均线）

    def refresh_white(self):
        super().refresh_white()

        allow, info = get_ma_index_indicator(
            symbol=self.white_index_symbol,
            period=self.white_ma_above_period,
        )
        if allow:
            t_white_codes = get_prefixes_stock_codes(self.white_prefixes)
            self.cache_whitelist.update(t_white_codes)


# 自定义前缀成份股 + 东方财富行业板块上涨比例预筛
class StocksPoolWhitePrefixesIndustry(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_prefixes = parameters.white_prefixes

    def refresh_white(self):
        super().refresh_white()

        section_names = get_dfcf_industry_sections()
        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name} 行业板块\n'
                f'{section_names}')
        t_white_codes = get_dfcf_industry_stock_codes(section_names)

        filter_codes = [code for code in t_white_codes if code[:2] in self.white_prefixes]
        self.cache_whitelist.update(filter_codes)


# 自定义前缀成份股 + 同花顺行业板块上涨个数预筛
class StocksPoolWhitePrefixesConcept(StocksPoolBlackWencai):
    def __init__(self, account_id: str, strategy_name: str, parameters, ding_messager):
        super().__init__(account_id, strategy_name, parameters, ding_messager)
        self.white_prefixes = parameters.white_prefixes

    def refresh_white(self):
        super().refresh_white()

        section_names = get_ths_concept_sections()
        if self.ding_messager is not None:
            self.ding_messager.send_text_as_md(
                f'[{self.account_id}]{self.strategy_name} 概念板块\n'
                f'{section_names}')
        t_white_codes = get_ths_concept_stock_codes(section_names)
        filter_codes = [code for code in t_white_codes if code[:2] in self.white_prefixes]
        self.cache_whitelist.update(filter_codes)
