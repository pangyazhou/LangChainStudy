"""
langchain异步调用大模型相关接口示例
"""
import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt_template = "给我讲一个关于{topic}的笑话"


# 异步调用 lcel版
def async_lcel():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    print(asyncio.run(chain.ainvoke("中国足球")))


# 异步流式输出
async def async_stream():
    prompt = ChatPromptTemplate.from_template(prompt_template)
    output_parser = StrOutputParser()
    chain = ({"topic": RunnablePassthrough()}
             | prompt
             | model
             | output_parser)
    async for msg in chain.astream("中国足球"):
        print(msg, end="", flush=True)


# 并行请求
def parallel_demo():
    chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    chain2 = (
            ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}")
            | model
    )
    combined = RunnableParallel(joke=chain1, poem=chain2)
    # print(chain1.invoke({"topic": "bears"}))
    # print(chain2.invoke({"topic": "bears"}))
    print(combined.invoke({"topic": "bears"}))


# 并行批量处理
def parallel_batch():
    chain1 = ChatPromptTemplate.from_template("tell me a joke about {topic}") | model
    chain2 = (
            ChatPromptTemplate.from_template("write a short (2 line) poem about {topic}")
            | model
    )
    combined = RunnableParallel(joke=chain1, poem=chain2)
    print(combined.batch([{"topic": "bears"}, {"topic": "cats"}]))


# 程序入口
if __name__ == "__main__":
    # async_lcel()
    asyncio.run(async_stream())
    # parallel_demo()
    # parallel_batch()