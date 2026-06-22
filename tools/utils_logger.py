import logging


def setup_logging(path=None, level=logging.DEBUG, file_line=False, silence_libs=True):
    """
    初始化日志配置
    :param path: 日志文件路径，None则输出到控制台
    :param level: 日志级别
    :param file_line: 是否显示文件名和行号
    :param silence_libs: 是否静默第三方库日志
    """
    file_line_fmt = ""
    if file_line:
        file_line_fmt = "%(filename)s[line:%(lineno)d] - %(levelname)s: "

    logging.basicConfig(
        level=level,
        format=file_line_fmt + "%(asctime)s|%(message)s",
        filename=path
    )

    # 静默常见第三方库的日志（完全忽略）
    if silence_libs:
        noisy_libs = [
            'gmtrade', 'gmtradelogger', 'tushare', 'urllib3', 'requests',
            'httpx', 'httpcore', 'asyncio', 'websockets',
            'apscheduler', 'schedule', 'chardet',
        ]
        # 既设置顶层 logger，也把已创建的同前缀子 logger 一并静默（例如 gmtrade.xxx）
        for lib in noisy_libs:
            logging.getLogger(lib).setLevel(logging.CRITICAL)
            logging.getLogger(lib).propagate = False

            for name, obj in logging.root.manager.loggerDict.items():
                if not isinstance(obj, logging.Logger):
                    continue
                if name == lib or name.startswith(lib + '.'):
                    obj.setLevel(logging.CRITICAL)  # 只输出崩溃级日志
                    obj.propagate = False  # 阻止传播到 root logger
