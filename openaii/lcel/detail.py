import asyncio
import os
from typing import Iterator
from typing import List
from concurrent.futures import ThreadPoolExecutor

import openai
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic


model = ChatOpenAI(model="gpt-3.5-turbo")
client = openai.OpenAI()
prompt_template = "给我讲一个关于{topic}的笑话"


def call_chat_model(messages: List[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    return response.choices[0].message.content


def invoke_chain(topic: str) -> str:
    prompt_value = prompt_template.format(topic=topic)
    messages = [{"role": "user", "content": prompt_value}]
    return call_chat_model(messages)


"""
接口调用
"""
# 原生调用
def simple():
    message = invoke_chain("中国足球")
    print(message)


# 使用CLEL调用
def chat_model():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    message = chain.invoke("中国足球")
    print(message)


"""
流式输出
"""
# 流式传输
def stream_chat_model(messages: List[dict]) -> Iterator[str]:
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    for response in stream:
        content = response.choices[0].delta.content
        if content is not None:
            yield content


def stream_chain(topic: str) -> Iterator[str]:
    prompt_value = prompt_template.format(topic=topic)
    return stream_chat_model([{"role": "user", "content": prompt_value}])


# 流式显示-原生
def stream_origin():
    for chunk in stream_chain("中国足球"):
        print(chunk, end="", flush=True)


# LCEL实现流式输出
def stream_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    for chunk in chain.stream("中国足球"):
        print(chunk, end="", flush=True)


"""
批量处理
"""
def batch_origin():
    result =  batch_chain(["中国足球", "苏联", "日本钢铁"])
    for msg in result:
        print(msg)


def batch_chain(topics: list) -> list:
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(invoke_chain, topics))


def batch_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    result = chain.batch(["中国足球", "苏联", "日本钢铁"])
    for msg in result:
        print(msg)


"""
异步
asyncio.run()
"""
def async_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    print(asyncio.run(chain.ainvoke("中国足球")))


"""
使用其他大模型
未获取app_key,暂时无法使用
"""
def model_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    anthropic = ChatAnthropic(model="claude-2")
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | anthropic
             | output_parser)
    print(chain.invoke("中国足球"))


"""
记录
基于LangSmith
需要api_key, 暂时无法使用
"""
def log_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print(chain.invoke("中国足球"))


"""
异步流
Async Stream
"""
async def async_stream():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    async for chunk in chain.astream("中国足球"):
        print(chunk, end="", flush=True)


# 程序入口
if __name__ == "__main__":
    # simple()
    # chat_model()
    # stream_origin()
    # stream_lcel()
    # batch_origin()
    # batch_lcel()
    # async_lcel()
    # model_lcel()
    # log_lcel()
    asyncio.run(async_stream())
