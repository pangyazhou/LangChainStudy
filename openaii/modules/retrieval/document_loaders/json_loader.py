"""
使用JSONLoader指定的jq 架构来解析 JSON 文件
"""

import json
from pathlib import Path


from langchain_community.document_loaders import JSONLoader


# 使用json加载
def json_module_invoke():
    file_path = 'example_data/facebook_chat.json'
    data = json.loads(Path(file_path).read_text())
    print(data)


# 使用JSONLoader 加载
# jq模块问题，代码尚未调通 TODO
def json_parser_invoke():
    loader = JSONLoader(
        file_path='./example_data/facebook_chat.json',
        jq_schema='.messages[].content',
        text_content=False)

    docs = loader.load()
    for doc in docs:
        print(doc)


# 程序入口
if __name__ == "__main__":
    # json_module_invoke()
    json_parser_invoke()
    pass
