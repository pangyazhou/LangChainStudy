import hashlib
import hmac
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time


# date参数生成规则
cur_time = datetime.now()
date = format_date_time(mktime(cur_time.timetuple()))
# 假使生成的date和下方使用的date = Fri, 05 May 2023 10:43:39 GMT


# authorization参数生成规则
tmp = "host: " + "spark-api.xf-yun.com" + "\n"
tmp += "date: " + date + "\n"
tmp += "GET " + "/v3.1/chat" + " HTTP/1.1"
"""上方拼接生成的tmp字符串如下
host: spark-api.xf-yun.com
date: Fri, 05 May 2023 10:43:39 GMT
GET /v1.1/chat HTTP/1.1
"""

tmp_sha = hmac.new(self.APISecret.encode('utf-8'), tmp.encode('utf-8'), 						digestmod=hashlib.sha256).digest()
"""此时生成的tmp_sha结果如下
b'\xcf\x98\x07v\xed\xe9\xc5Ux\x0032\x93\x8e\xbb\xc0\xe5\x83C\xda\xba\x05\x0c\xd1\xdew\xccN7?\r\xa4'
"""