# -*- coding:utf-8 -*-
import hashlib
import base64
import hmac
import time
import json
import websocket
import _thread as thread
import ssl

class Document_Q_And_A:
    def __init__(self, APPId, APISecret, TimeStamp, OriginUrl):
        self.appId = APPId
        self.apiSecret = APISecret
        self.timeStamp = TimeStamp
        self.originUrl = OriginUrl

    def get_origin_signature(self):
        m2 = hashlib.md5()
        data = bytes(self.appId + self.timeStamp, encoding="utf-8")
        m2.update(data)
        checkSum = m2.hexdigest()
        return checkSum



    def get_signature(self):
        # 获取原始签名
        signature_origin = self.get_origin_signature()
        # print(signature_origin)
        # 使用加密键加密文本
        signature = hmac.new(self.apiSecret.encode('utf-8'), signature_origin.encode('utf-8'),
                             digestmod=hashlib.sha1).digest()
        # base64密文编码
        signature = base64.b64encode(signature).decode(encoding='utf-8')
        # print(signature)
        return signature



    def get_header(self):
        signature = self.get_signature()
        header = {
            "Content-Type": "application/json",
            "appId": self.appId,
            "timestamp": self.timeStamp,
            "signature": signature
        }
        return header

    def get_url(self):
        signature = self.get_signature()
        header = {
            "appId": self.appId,
            "timestamp": self.timeStamp,
            "signature": signature
        }
        return self.originUrl + "?" + f'appId={self.appId}&timestamp={self.timeStamp}&signature={signature}'
        # 使用urlencode会导致签名乱码
        # return self.originUrl + "?" + urlencode(header)



    def get_body(self):
        data = {
            "chatExtends": {
                "wikiPromptTpl": "请将以下内容作为已知信息：\n<wikicontent>\n请根据以上内容回答用户的问题。\n问题:<wikiquestion>\n回答:",
                "wikiFilterScore": 0.83,
                "temperature": 0.5
            },
            "fileIds": [
                "{上传upload接口返回的fileId}"
            ],
            "messages": [
                {
                    "role": "user",
                    "content": "父亲在车站买了什么东西？"
                }
            ]
        }
        return data

# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws, close_status_code, close_msg):
    print("### closed ###")
    print("关闭代码：", close_status_code)
    print("关闭原因：", close_msg)


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(ws.question)
    ws.send(data)

# 收到websocket消息的处理
def on_message(ws, message):
    # print(message)
    data = json.loads(message)
    code = data['code']
    if code != 0:
        print(f'请求错误: {code}, {data}')
        ws.close()
    else:
        content = data["content"]
        status = data["status"]
        # print(f'status = {status}')
        print(content, end='')
        if status == 2:
            ws.close()


if __name__ == '__main__':
    # 先去 开放平台控制台（https://console.xfyun.cn）创建应用，获取下列应用信息进行替换
    APPId = "******"
    APISecret = "******"

    curTime = str(int(time.time()))
    OriginUrl = "wss://chatdoc.xfyun.cn/openaii/chat"
    document_Q_And_A = Document_Q_And_A(APPId, APISecret, curTime, OriginUrl)

    wsUrl = document_Q_And_A.get_url()
    print(wsUrl)
    headers = document_Q_And_A.get_header()
    body = document_Q_And_A.get_body()

    # 禁用WebSocket库的跟踪功能，使其不再输出详细的调试信息。
    websocket.enableTrace(False)
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = APPId
    ws.question = body
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})


    # 文档问答成功