"""
ChatModels是LangChain的核心组件。
LangChain不提供自己的ChatModels，而是提供与许多不同模型交互的标准接口。
具体来说，该接口将消息列表作为输入并返回消息

1. How to cache ChatModel responses
2. How to use ChatModels that support function calling
3. How to stream responses from a ChatModel
4. How to track token usage in a ChatModel call
"""
import asyncio

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# 语言模型对象
chat = ChatOpenAI()

"""
聊天模型界面基于消息而不是原始文本。
LangChain目前支持的消息类型有AIMessage, HumanMessage, SystemMessage,FunctionMessage和ChatMessage- ChatMessage接受任意角色参数。
大多数时候，您只需处理HumanMessage、AIMessage和 SystemMessage
"""
def chatmodel_invoke():
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content="What is the purpose of model regularization?"),
    ]
    # 正常调用
    result = chat.invoke(messages)
    print(result)
    # 流式输出
    for chunck in chat.stream(messages):
        print(chunck.content, end="", flush=True)
    # 批量调用
    result = chat.batch([messages])
    print(result)
    # 异步调用
    print(asyncio.run(chat.ainvoke(messages)))
    # 异步流式输出日志
    asyncio.run(async_stream_log(messages))



async def async_stream_log(messages):
    async for chunk in chat.astream_log(messages):
        print(chunk)


# 程序入口
if __name__ == "__main__":
    chatmodel_invoke()
    pass