"""
将本文件复制重命名为 credentials.py 生效
"""

USE_BIG_QMT = False
# True：交易/行情走项目内 btquant（大 QMT 桥接，需 qmt_bridge.json + BIGQMTHELPER）
# False：走券商 miniqmt 原生 xtquant（需 QMT_CLIENT_PATH）
# 注意：DailyHistoryXT 历史日线下载仅支持原生 xtquant，USE_BIG_QMT=True 时请改用 Tushare/MOOTDX

# ================ 本地缓存 ================
CACHE_PROD_PATH = './_cache/prod_pwc'   # 生本地缓存目录：生产环境（一般用于实盘）
CACHE_TEST_PATH = './_cache/test_pwc'   # 本地缓存目录：测试环境（一般用于模拟盘）

# ================= 交易账号 =================
# 具体账号信息请咨询对应券商客服经理
QMT_ACCOUNT_ID = '55009728'
QMT_CLIENT_PATH = r'C:\国金证券QMT交易端\userdata_mini'

# ================= 策略通知 =================
# 申请方式见文档：https://github.com/silver6wings/SilverQuant?tab=readme-ov-file#%E7%94%B3%E8%AF%B7%E9%92%89%E9%92%89%E6%9C%BA%E5%99%A8%E4%BA%BA
DING_SECRET = 'SECa0ab7f3ba9742c0*********'
DING_TOKENS = 'https://oapi.dingtalk.com/robot/send?access_token=**********************'

# 申请方式见文档：https://github.com/silver6wings/SilverQuant?tab=readme-ov-file#%E7%94%B3%E8%AF%B7%E6%8E%98%E9%87%91%E6%A8%A1%E6%8B%9F%E7%9B%98
GM_CLIENT_TOKEN = 'ad239ba1e307c4e5f31fce19a6c173fb********'
GM_ACCOUNT_ID = '189ca421-49db-11ef-9fa8-0016********'

# ================= Amazing NATS 行情 =================
# 启动 producer 后，本仓库通过 NATS 接收 QMT 格式 tick
# 以下两项须与 sickle credentials 中 NATS_PRODUCER_URL / NATS_PRODUCER_SUBJECT 一致
NATS_CONSUMER_URL = 'nats://127.0.0.1:4222'
NATS_CONSUMER_SUBJECT = 'market.tick.amazing'

# ================= 远程推送 =================
# 需自行搭建服务，用于推送选股结果
RECOMMEND_HOST = 'http://127.0.0.1:5000'
AUTHENTICATION = '*********************'

# ================= 自动脚本 =================
LAUNCHER_SCRIPTS = 'run_remote.py'

# ================= 数据源 =================
# 本地通达信安装目录，用以mootdx数据加速以及访问通达信自选列表
TDX_FOLDER = ''  # r'C:\new_tdx'

# mootdx 在线行情：运行 `python -m mootdx bestip -vv` 可测可用 IP
# MOOTDX_SERVER 优先于内置列表；留空则依次尝试内置节点
MOOTDX_SERVER = None  # ('202.108.253.139', 80)
# True 时每次启动全量 bestip 测速（慢，且部分环境会卡住）
MOOTDX_USE_BESTIP = False
# 长驻进程可设 True 维持心跳；一次性脚本请保持 False，否则进程可能无法退出
MOOTDX_HEARTBEAT = False

# 可以设置多个token
TUSHARE_TOKEN = [
    ['(your token)', '(your custom token name)'],
]
