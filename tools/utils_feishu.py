import base64
import hashlib
import hmac
import json
import requests
import time
import traceback
from typing import Dict, Any

from tools.constants import MSG_INNER_SEPARATOR, MSG_OUTER_SEPARATOR
from tools.utils_ding import BaseMessager


def get_feishu_markdown_card(title, text):
    """
    生成飞书富文本卡片 (JSON 2.0 结构)
    """
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
                    "text_align": "left",
                    "text_size": "normal_v2",
                    "margin": "0px 0px 0px 0px"
                }
            ]
        },
        "header": {
            "title": {
                "tag": "plain_text",
                "content": title
            },
            "template": "blue",
            "padding": "12px 12px 12px 12px"
        }
    }
    return card_style


class FeishuMessager(BaseMessager):
    def __init__(self, secret: str = None, webhook_url: str = None):
        """
        https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot?lang=zh-CN
        :param secret: 安全设置的签名
        :param url: 机器人的WebHook_url
        """
        self.secret = secret
        self.webhook_url = webhook_url
        self.refresh_webhook()

    def refresh_webhook(self) -> bool:
        """检查配置是否完整，并在缺少时打印提示"""
        if self.secret is None or self.webhook_url is None:
            print('--- FeishuMessager 配置提示 ---')
            if self.secret is None:
                print('请先在飞书申请secret')
                print('格式:SECa0ab7f3ba9742c0*********')
            if self.webhook_url is None:
                print('请先在飞书申请webhook')
                print('格式:https://open.feishu.cn/open-apis/bot/v2/hook/****************')
            print('------------------------------------')
            return False
        return True

    def gen_sign(self, timestamp: int) -> str:
        """
        生成签名
        :param timestamp: 时间戳 (必须与消息体中的 timestamp 一致)
        :return: sign
        """
        # 拼接 timestamp 和 secret
        string_to_sign = '{}\n{}'.format(timestamp, self.secret)
        hmac_code = hmac.new(string_to_sign.encode(
            "utf-8"), digestmod=hashlib.sha256).digest()
        # 对结果进行 Base64 处理
        sign = base64.b64encode(hmac_code).decode('utf-8')
        return sign

    def send_message(self, data: Dict[str, Any]) -> dict:
        """
        发送消息至机器人对应的群
        :param data: 发送的内容
        :return: 飞书服务器的响应字典
        """
        try:
            if not self.refresh_webhook():
                return {'msg': 'ConfigurationError', 'code': -1}

            header = {
                "Content-Type": "application/json",
                "Charset": "UTF-8"
            }

            send_data = json.dumps(data)
            send_data = send_data.encode("utf-8")

            response = requests.post(
                url=self.webhook_url, data=send_data, headers=header)
            return json.loads(response.text)
        except Exception:
            traceback.print_exc()
            return {'msg': 'Exception!', 'code': -99}

    def send_text(self, text: str, output: str = '', alert: bool = False) -> bool:
        """发送普通文本消息"""
        timestamp = round(time.time())
        sign = self.gen_sign(timestamp)

        content_elements = [
            {
                "tag": "text",
                "text": text,
            }
        ]
        if alert:
            content_elements.append({
                "tag": "at",
                "user_id": "all"
            })

        res = self.send_message(data={
            "timestamp": timestamp,
            "sign": sign,
            "msg_type": "post",
            "content": {"post":
                {"zh_cn":
                    {
                        "title": "",
                        "content": [content_elements]
                    }
                }
            }
        }
        )



        #{'StatusCode': 0, 'StatusMessage': 'success', 'code': 0, 'data': {}, 'msg': 'success'}

        if res.get('code') == 0 or res.get('StatusCode') == 0:
            if len(output) > 0:
                print(output, end='')
            else:
                print('Feishu message send success!')
            return True
        else:
            print('Feishu message send failed: ', {res})
            return False

    def send_text_as_md(self, text: str, output: str = '', alert: bool = False) -> bool:
        """将多行文本格式化为 Markdown 引用样式发送"""
        title = text.split('\n')[0]
        text = text.replace(MSG_OUTER_SEPARATOR, '\n\n>')
        text = text.replace(MSG_INNER_SEPARATOR, '\n')
        return self.send_markdown(title, text, output, alert)

    def send_markdown(self, title: str, text: str, output: str = '', alert: bool = False) -> bool:
        """发送 Markdown (交互式卡片) 消息"""
        #飞书 markdown 颜色替换
        color_replace_dic = {"#DC2832": "red", "#16BC50": "green"}
        for a, b in color_replace_dic.items():
            text = text.replace(a, b)
            
        text += "\n<at id=all></at>" if alert else ""
        timestamp = round(time.time())
        sign = self.gen_sign(timestamp)

        my_data = {
            "timestamp": timestamp,
            "sign": sign,
            'msg_type': 'interactive',
            "card": get_feishu_markdown_card(title, text)
        }

        res = self.send_message(data=my_data)
        if res.get('code') == 0 or res.get('StatusCode') == 0:
            if len(output) > 0:
                print(output, end='')
            else:
                print('Feishu markdown send success!')
            return True
        else:
            print(f'Feishu markdown send failed: {res}')
            return False