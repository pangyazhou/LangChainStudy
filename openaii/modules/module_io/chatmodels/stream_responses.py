"""
How to stream responses from a ChatModel
正常流式输出
"""
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")


def stream_response_invoke():
    for chunk in model.stream("给我写一个关于我的县长父亲的散文，字数在400字左右"):
        print(chunk.content, end="", flush=True)


# 程序入口
if __name__ == "__main__":
    stream_response_invoke()
    pass