import base64
import hashlib
import hmac
import json
import time
import requests
import traceback
import urllib.parse
import urllib.request
from datetime import datetime


def get_markdown_card(title, text):
    card_style = {
        "schema": "2.0",
        "config": {
            "update_multi": True,
            "style": {
                "text_size": {
                    "normal_v2": {
                        "default": "normal",
                        "pc": "normal",
                        "mobile": "heading"
                    }
                }
            }
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": [
                {
                    "tag": "markdown",
                    "content": text,
                    # "content": "飞书emoji :OK::THUMBSUP:\n*斜体* **粗体** ~~删除线~~ \n<font color='red'>这是红色文本</font>\n<text_tag color='blue'>标签</text_tag>\n<number_tag>1</number_tag>\n[文字链接](https://open.feishu.cn/server-docs/im-v1/message-reaction/emojis-introduce)\n<link icon='chat_outlined' url='https://open.feishu.cn' pc_url='' ios_url='' android_url=''>带图标的链接</link>\n<at id=all></at>\n- 无序列表1\n    - 无序列表 1.1\n- 无序列表2\n1. 有序列表1\n    1. 有序列表 1.1\n2. 有序列表2\n```JSON\n{\"This is\": \"JSON demo\"}\n```\n`inline-code`\n# 一级标题\n## 二级标题\n> 这是一段引用\n\n | Syntax | Description |\n| -------- | -------- |\n| Header | Title |\n| Paragraph | Text |",
                    "text_align": "left",
                    "text_size": "normal_v2",
                    "margin": "0px 0px 0px 0px"
                }
                # {
                #     "tag": "hr",
                #     "margin": "0px 0px 0px 0px"
                # }
            ]
        },
        "header": {
            "title": {
                "tag": "plain_text",
                "content": title
            },
            # "subtitle": {
            #     "tag": "plain_text",
            #     "content": text
            # },
            "template": "blue",
            "padding": "12px 12px 12px 12px"
        }
    }
    return card_style


class FeishuMessager(object):
    def __init__(self, secret: str = None, webhook_url: str = None):
        """
        https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot?lang=zh-CN
        :param secret: 安全设置的签名
        :param url: 机器人的WebHook_url
        """
        self.secret = secret
        self.webhook_url = webhook_url
        self.refresh_webhook()

    def refresh_webhook(self):
        if self.secret is None or self.webhook_url is None:
            print('请先在飞书申请secret')
            print('格式:SECa0ab7f3ba9742c0*********')
            print('请先在飞书申请webhook')
            print(
                '格式:https://open.feishu.cn/open-apis/bot/v2/hook/****************')
            return False
        return True

    def gen_sign(self, timestamp, secret):
        # 拼接 timestamp 和 secret
        # timestamp = round(time.time())  # 时间戳
        timestamp = int(datetime.now().timestamp())
        string_to_sign = '{}\n{}'.format(timestamp, secret)
        hmac_code = hmac.new(string_to_sign.encode(
            "utf-8"), digestmod=hashlib.sha256).digest()
        # 对结果进行 Base64 处理
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return sign

    def send_message(self, data) -> dict:
        # """
        # 发送消息至机器人对应的群
        # :param data: 发送的内容
        # :return:
        # """
        try:
            if self.refresh_webhook():
                header = {
                    "Content-Type": "application/json",
                    "Charset": "UTF-8"
                }

                send_data = json.dumps(data)
                send_data = send_data.encode("utf-8")

                response = requests.post(
                    url=self.webhook_url, data=send_data, headers=header)
                return json.loads(response.text)
        except:
            traceback.print_exc()
            return {'msg': 'Exception!'}

    def send_text(self, text: str, output: str = '', alert: bool = False) -> bool:

        timestamp = round(time.time())
        sign = self.gen_sign(timestamp, self.secret)
        res = self.send_message(data={
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "post",
            "content": {"post":
                        {"zh_cn":
                         {
                             "title": "",
                             "content": [
                                 [
                                     {
                                         "tag": "text",
                                         "text": text,
                                     }
                                 ] + (
                                     [
                                         {
                                             "tag": "at",
                                             "user_id": "all"
                                         }
                                     ] if alert else []
                                 )
                             ]
                         }
                         }
                        }
        }
        )

        if res['msg'] == 'success':
            if len(output) > 0:
                print(output, end='')
            else:
                print('Feishu message send success!')
            return True
        else:
            print('Feishu message send failed: ', res['msg'])
            return False

    def send_text_as_md(self, text: str, output: str = '', alert: bool = False) -> bool:
        title = text.split('\n')[0]
        text = text.replace('\n', '\n>\n>')

        return self.send_markdown(title, text, output, alert)

    def send_markdown(self, title: str, text: str, output: str = '', alert: bool = False) -> bool:
        # my_data = {
        #     "msgtype": "markdown",
        #     "markdown": {
        #         "title": "测试markdown样式",
        #         "text": "# 一级标题 \n## 二级标题 \n> 引用文本  \n**加粗**  \n*斜体*  \n[百度链接](https://www.baidu.com) "
        #             "\n![草莓](https://dss0.bdstatic.com/70cFuHSh_Q1YnxGkpoWK1HF6hhy/it/u=1906469856,4113625838&fm=26&gp=0.jpg)"
        #             "\n- 无序列表 \n1.有序列表  \n@某手机号主 @18688889999"},
        #     "at": {
        #         "atMobiles": [""],
        #         "isAtAll": False}  # 是否@所有人
        # }
        text += "\n<at id=all></at>" if alert else ""
        timestamp = round(time.time())
        sign = self.gen_sign(timestamp, self.secret)
        my_data = {
            "timestamp": timestamp,
            "sign": sign,
            'msg_type': 'interactive',
            "card": get_markdown_card(title, text)
        }
        res = self.send_message(data=my_data)
        if res['msg'] == 'success':
            if len(output) > 0:
                print(output, end='')
            else:
                print('Feishu markdown send success!')
            return True
        else:
            print('Feishu markdown send failed: ', res['msg'])
            return False
