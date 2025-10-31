> 成功的秘诀有三条：
>
> 第一，尽量避免风险，保住本金
>
> 第二，尽量避免风险，保住本金
>
> 第三，坚决牢记第一、二条。
>
> —— 沃伦·爱德华·巴菲特 Warren Edward Buffett

---

来跟我大声读三遍：先轻仓模拟。先轻仓模拟？先轻仓模拟！🤤

提问前先看下我辛苦写的说明好不啦，蠢尚可救，懒无药医！🙄

如果觉得好用就给个星星反馈一下，好让作者有动力更新吖！😘

目前暂未建立任何付费分享渠道，请明辨真假，谨防冒牌货！🥸

Github 因某些原因不便境内服务器部署，这里有：[国内镜像](https://gitee.com/silver6wings/silverQuant)

# 鸣谢

* 感谢 [@owen590](https://github.com/owen590) 从第一行代码开始至今提供的宝贵意见 
* 感谢 [@dominicx](https://github.com/dominicx) 提交的PR，修复作者未覆盖的交易场景细节
* 感谢 [@nackel](https://github.com/nackel) 提交的PR，添加飞书机器人模块的支持
* 感谢 [@vipally](https://github.com/vipally) 提交的PR，优化部分MyTT公式和vscode配置

# 项目简介

SilverQuant 是基于 [MiniQMT](https://dict.thinktrader.net/nativeApi/start_now.html)
开发的A股证券🇨🇳全自动交易框架，开箱即可运行

旨在帮助新进入量化领域的同学解决大部分技术启动问题，支持在本地执行策略

组件化设计可以快速根据策略思路搭建原型并建立模拟测试，还能一键切换实盘

术的层面有千百门战法，回归道的层面，交易的游戏总结起来就只有四个关卡：

1. 保住本金不大幅亏损
2. 可以实现阶段性获利
3. 能够稳定持续的盈利
4. 尽可能地提高利润率

本框架赠予您一套短线防弹衣，避免初入市场即被深度套牢，帮您闯过第一关

之后的道路，还需各位少侠自渡，时常自省，踏实自悟，各有渡口，各乘归舟

是修行的路，也注定是凶险的路，起落有时，顺逆皆安，岁月沉香，星河璀璨

希望您也可以在修罗场里的磨砺中，能打造出属于自己合适的交易模式，共勉

# 设计规格

> `release_x.x.x`分支为稳定发布版本；
> 
> `main` 分支为最近待发布版本;
>
> `develop` 分支为开发版本，包含所有最近的更新；

以下为架构示意图，帮助二次开发的朋友们更好地理解和快速上手

![image](_imgs/architecture.png)

数据源

> 历史数据源：主要使用第三方`AKShare`, `Tushare`, `TDX` 以及 `MiniQMT` 自带的数据源

> 盘中数据源：主要使用 `MiniQMT` 提供的 Tick 级别的数据源

## 策略

> 策略可以设置执行频率，秒级处理最新的截面数据，订阅范围可以自行定义

## 交易端

> 交易端可以对接 QMT 实盘和掘金的模拟盘，支持市价单、限价单以及撤单

> 目前暂无TWAP，VWAP以及冰山指令等拆单算法


# 快速开始

## 系统需求

> 至少是 Windows 系统，国内交易软件生态大都不支持 Mac 和 Linux

## 软件下载

> 下载 Github 桌面版
> 
> https://desktop.github.com/

> 下载 PyCharm CE for Windows（注意是社区免费版即可，不必需Professional）
> 
> https://www.jetbrains.com/pycharm/download/?section=windows
> 
> *注：[VSCode](https://code.visualstudio.com/) 因为配置过程更复杂，所以不做新手推荐*

> 下载对应的券商版 QMT（需要提前找券商客户经理开通账户的 QMT 权限）
> 
> 截至2024年底，支持QMT的券商可以在[这里](https://www.bilibili.com/opus/1014402646051651589)查询，以[某券商](https://miniqmt.com/)为例：
> 
> WinRAR下载
> 
> https://www.win-rar.com/start.html?&L=0
> 
> https://www.rarlab.com/download.htm
> 
> 某券商版 QMT 实盘
> 
> https://download.gjzq.com.cn/gjty/organ/gjzqqmt.rar
> 
> 某券商版 QMT 模拟
> 
> https://download.gjzq.com.cn/temp/organ/gjzqqmt_ceshi.rar

> 如果需要问财大模型相关功能，需要下载 Node.JS 版本 v16+
>
> https://nodejs.org/zh-cn

> 如果需要模拟盘先做策略测试，需要下载 掘金3 的仿真交易软件
> 
> https://www.myquant.cn/terminal


## 配置环境

克隆项目到本地

> 可以直接在Github Desktop里克隆
> 
> 也可以直接在终端输入 `gh repo clone silver6wings/SilverQuant` 然后在Github Desktop里打开

用 PyCharm 打开刚才 clone 的 SilverQuant 文件夹：主菜单 File > Open，选择 SilverQuant 文件夹

安装 Python 3.10 版本（作者研发用的稳定版本，不强求但用这个潜在问题一定最少）
 
> 1. 可以打开PyCharm在IDE里安装
> 2. 也可以直接去Python官网下载：https://www.python.org/downloads/release/python-31010/

安装对应的包
 
> 在PyCharm里安装依赖，打开终端(Terminal)输入：`pip install -r requirements.txt`
> 
> *注：如果事先已经打开了终端，需要关闭并重新打开终端看到类似 `(venv)` 之后再执行上述命令*

如果安装慢可以使用镜像源，在指令后加 ` -i [镜像网址]` 

> 可用的镜像网址
> 
> * https://pypi.tuna.tsinghua.edu.cn/simple/
> * https://pypi.mirrors.ustc.edu.cn/simple/
> * http://pypi.mirrors.ustc.edu.cn/simple/
> * http://mirrors.aliyun.com/pypi/simple/

## 启动程序

### 启动QMT交易软件

启动券商版迅投QMT的（勾选）`极简模式`，最新版的为`独立交易`，确认左下角数据源的链接状态正常

### 配置 Credentials

> 复制项目根目录下的`credentials_sample.py`，改名为`credential.py`并填入自己的参数
> 
> 1. `AUTHENTICATION` 是远程策略获取推送服务的密钥，其他策略不需要可置空
> 2. `CACHE_BASE_PATH` 是本地策略缓存文件夹路径，可不用修改
> 3. `QMT_XXX`的两项是账户相关，需要股票账户id和QMT安装位置，找不到`userdata_mini`文件夹需要先运行QMT一次
> 4. `DING_XXX`的两项是群通知相关，钉钉通知需要建群，然后建立机器人获取 Webhook URL
> 5. `GM_XXX`的两项是模拟盘相关，模拟盘需要自行获取掘金的 Secret Tokens

### 申请钉钉机器人

> 如果需要钉钉机器人播报，可以自行建通知群，首先拉至少三人创建一个普通群
>
> 1. 群设置里：机器人 -> 添加机器人 -> 自定义机器人 -> 添加
> 2. 安全设置：加签 获取 `DING_SECRET`
> 3. 勾选“同意xxxx”，并下一步
> 4. Webhook栏，点击复制，获取 `DING_TOKENS`
> 5. 配置到`credentials.py`

### 申请掘金模拟盘

> 如果需要模拟盘测试，需要先下载安装掘金3客户端，链接见上文
> 
> 1. 新用户可以先拿手机号注册一个新的账号
> 2. 在`系统设置`中找到 密钥管理(Token) 即 `GM_CLIENT_TOKEN`
> 3. 在`账户管理`中添加仿真账户，并完成模拟入金和模拟交易费率设置
> 4. 在账户管理中找到`复制账户ID`，获取 `GM_ACCOUNT_ID`
> 5. 配置到`credentials.py`

### 启动脚本

> PyCharm 中打开 SilverQuant 项目根目录
> 
> 1. 找到 `run_xxxxxx.py` 文件，根目录下
> 2. 找到 `IS_PROD = False` 代码处，将 False 改为 True 切换成实盘
> 3. 确认 QMT 交易软件正在运行
> 4. 启动 `run_xxxxxx.py` 文件，点击绿色小三角

# 入口说明

策略本身有一些开箱即用的启动程序，可以直接运行。

由于 QMT 针对单个操作系统 instance 只能单开，所以每次尽可能只 run 一个策略，以防冲突。

```
run_wencai_qmt.py
利用同花顺问财大模型选股买入，
需自行定义prompt选股问句，在`selector/select_wencai.py`中定义
可以用来快速建立原型，做模拟盘测试，预估大致收益，一般至少测试一个月
如果需要复杂卖出策略，需要参考`run_remote.py`加入下载历史数据代码

run_wencai_tdx.py
使用问财和通达信的数据源，可以脱离QMT客户端运行模拟盘
```
```
run_remote.py
对于需要Linux或者分布式大数据计算的场景
可以自行搭建推票服务，程序会通过http协议远程读取数据执行买入
```
```
run_shield.py
适用于手动买入，量化卖出的半自动场景需求
```
```
run_swords.py
票池打板的模板，当封板且满足封板要求的时候进行买入，需要优化
```
```
run_swords_tdx.py
票池打板的模板，同上，只不过票池是读取本地通达信的自选股列表
```
```
run_ai_gen.py
适用于进阶学习，公式选股的样板，请勿用于实盘，因为确实不赚钱
```

# 高级配置

> 查看进阶配置文档: [[CONFIGURATION]](_doc/CONFIGURATION.md)

> 查看完整变更历史: [[CHANGELOG]](_doc/CHANGELOG.md)

# 已知问题

> 开仓价从`Mini QMT`直接获取，目前不会根据除权情况动态调整，可能会因为除权价格降低导致非正常止损卖出。
> 
> Akshare & Pywencai 等库由于各大网站限流策略，经常被Ban IP，注意数据量控制问题。
> 
> 使用 Tushare 的历史日线数据源有不复权的问题，使用历史数据时注意tick的lastClose与昨日收盘不一致问题。
> 
> ETF 由于日线库不全所以会有日线相关指标卖出策略失效，请注意测试时这种情况不报错。

# 常见问题 Q & A

About QMT
```
如果出现程序在控制台没有持续输出，需要在QMT中检查行情源是否正确设置
```

About setup
```
启动之前最好重启一下系统刷新所有的软件配置
```

About node.js

```
如果出现类似如下错误，说明在使用pywencai的过程中node版本过高，非严重问题
可以通过降低版本，或者配置系统环境变量 NODE_NO_WARNINGS=1 解决报警输出

(node:44993) [DEP0040] DeprecationWarning: The `punycode` module is deprecated. Please use a userland alternative instead.
(Use `node --trace-deprecation ...` to show where the warning was created)
```

About pywencai
``` 
pywencai的原理是去 https://www.iwencai.com/ 抓取数据，
记得一定要先安装 Node.js，安装完毕至少要重启PyCharm一次
否则会报错：'NoneType' object has no attribute 'get' 
其次检查自己的选股提示词 (Prompt) 能不能在网页上搜到票
最后，间隔建议设置为30秒或者更长，否则容易被封IP
$ pip install pywencai --upgrade 
```

About akshare
```
akshare 会去各个官方网站抓取公开数据，网站改版会导致爬虫失效
akshare 更新比较及时，升级 akshare 版本到最新会解决一些问题
$ pip install akshare --upgrade
```

About tushare

可以在 tushare 的官网 [tushare.pro](https://tushare.pro/register?reg=430410) 注册获得 token
```
tushare 作为 akshare 的备用数据源，需要配置对应的 token 才可以使用
框架支持多个 token，请参考 reader/tushare_token_sample.py 
```

About mytt

框架内置 [MyTT](https://github.com/mpquant/MyTT) 的部分代码
```
同时针对部分核心函数不支持动态参数的局限添加改进后的强化版本
```

---

# 免责声明

本项目遵循 Apache 2.0 开原许可协议

* 财不入急门，强烈建议在您的策略被验证成熟之前，至少先轻仓实盘谨慎测试
* 对于代码使用过程中造成的任何损失，作者不承担任何责任，请注意合规使用
* 对于代码改进有任何想法和建议，欢迎在 [Issues](https://github.com/silver6wings/SilverQuant/issues) 提交问题或直接提交PR修复

# 关于作者

本作者是一位在外企打工多年的牛马，未曾供职过任何金融机构

量化只是业余兴趣爱好，项目一定有不少瑕疵待完善，还请谅解

使用过程中若遇到任何深度的问题，资源或项目有意向寻求合作

带着你的Star和Github的ID添加作者的WX工作号: `junchaoyu_`

这里有卧虎藏龙的技术支持群讨论各种问题，欢迎各路英雄好汉
