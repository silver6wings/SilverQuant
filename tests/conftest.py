# -*- coding: utf-8 -*-
import shutil
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_ROOT = next((p for p in (_here, *_here.parents) if (p / "tools" / "utils_cache_ak.py").is_file()), _here.parent)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "local_only: 仅本地跑")
    from tools.utils_cache_ak import TRADE_DAY_CACHE_PATH

    c = Path(TRADE_DAY_CACHE_PATH)
    if c.is_file():
        return
    leg = _ROOT / "_cache" / "_open_day_list_sina.csv"
    if leg.is_file():
        c.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(leg, c)
        return
    ds = "2025-09-29,2025-09-30,2025-10-09,2025-10-10,2025-10-13,2025-10-14,2025-10-15,2025-10-16,2025-10-17,2025-10-20,2025-10-21,2025-10-22".split(",")
    c.parent.mkdir(parents=True, exist_ok=True)
    c.write_text(",trade_date\n" + "\n".join(f"{i},{d}" for i, d in enumerate(ds)) + "\n", encoding="utf-8")
