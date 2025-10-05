# 将本文件复制重命名为 credentials.py 生效

CACHE_PROD_PATH = './_cache/prod_pwc'   # 生产环境缓存目录
CACHE_TEST_PATH = './_cache/test_pwc'   # 测试环境缓存目录

# 本地通达信安装目录，用以mootdx数据加速以及访问通达信自选列表
TDX_FOLDER = r'C:\new_tdx'
TUSHARE_TOKEN = [
    # 可以设置多个token
    ['(your token)', '(your label)'],
]

# 具体账号信息请咨询对应券商客服经理
QMT_ACCOUNT_ID = '55009728'
QMT_CLIENT_PATH = r'C:\国金证券QMT交易端\userdata_mini'

# 申请方式见文档：https://github.com/silver6wings/SilverQuant?tab=readme-ov-file#%E7%94%B3%E8%AF%B7%E9%92%89%E9%92%89%E6%9C%BA%E5%99%A8%E4%BA%BA
DING_SECRET = 'SECa0ab7f3ba9742c0*********'
DING_TOKENS = 'https://oapi.dingtalk.com/robot/send?access_token=**********************'

# 申请方式见文档：https://github.com/silver6wings/SilverQuant?tab=readme-ov-file#%E7%94%B3%E8%AF%B7%E6%8E%98%E9%87%91%E6%A8%A1%E6%8B%9F%E7%9B%98
GM_CLIENT_TOKEN = 'ad239ba1e307c4e5f31fce19a6c173fb********'
GM_ACCOUNT_ID = '189ca421-49db-11ef-9fa8-0016********'

# 需自行搭建服务，用于推送选股结果
RECOMMEND_HOST = 'http://127.0.0.1:5000'
AUTHENTICATION = '*********************'
