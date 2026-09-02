# -*- coding: utf-8 -*-
"""
btquant 影子包（大 QMT 桥接）。

本包不依赖 miniqmt 原生 xtquant 库，而是把 SilverQuant 对 xtquant 的调用
代理到大 QMT helper 的 HTTP 网关（默认 127.0.0.1:9000）。

启用方式：credentials.py 中 USE_BIG_QMT=True，业务代码通过 tools.utils_xtquant 导入。

配置文件：项目根目录 qmt_bridge.json。说明见 _doc/BIG_QMT.md。
"""
from . import xtconstant
from . import xttype
from . import xtdata
from . import xttrader

__all__ = ["xtconstant", "xttype", "xtdata", "xttrader"]
