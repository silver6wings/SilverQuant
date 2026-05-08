# -*- coding: utf-8 -*-
"""testcase/tests 公共配置：项目根路径与交易日历 CSV 就绪。"""
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    """注册 markers；若标准路径无交易日 CSV 而仓库存在旧路径文件则复制，避免读盘失败。"""
    config.addinivalue_line("markers", "local_only: 依赖本地数据或网络的用例，默认不跑")
    from tools.utils_cache_ak import TRADE_DAY_CACHE_PATH

    canonical = Path(TRADE_DAY_CACHE_PATH)
    if canonical.is_file():
        return
    legacy = _ROOT / "_cache" / "_open_day_list_sina.csv"
    if legacy.is_file():
        canonical.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(legacy, canonical)
